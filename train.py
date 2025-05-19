# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from llava.eval.my_llava import *
from llava.mm_utils import (get_model_name_from_path, tokenizer_image_token_batch)

IMAGE_TOKEN_INDEX = -200


def get_train_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="./liuhaotian/llava-v1.6-vicuna-7b")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--conv_mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=4096)

    parser.add_argument("--data_root", type=str, default='/data')
    parser.add_argument("--set_name", type=str, default='bo')
    parser.add_argument("--prefix_token", type=int, default=4)
    parser.add_argument("--flip_p", type=float, default=0.5)
    parser.add_argument("--user_prompt", default=False, action='store_true')
    parser.add_argument("--recog_only", default=True, action='store_true')

    parser.add_argument("--tensorboard_path", type=str, default='./runs/')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoints/')
    parser.add_argument("--exp_name", type=str, default='./debug/')
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=10)
    train_args = parser.parse_args()
    return train_args


if __name__ == "__main__":

    args = get_train_args()

    os.makedirs(f"./checkpoints/{args.set_name}/", exist_ok=True)

    writer = SummaryWriter(os.path.join(args.tensorboard_path, args.set_name, args.exp_name))
    save_location = os.path.join(args.checkpoint_path, args.set_name, args.exp_name)
    os.makedirs(save_location, exist_ok=True)
    args.model_name = get_model_name_from_path(args.model_path)

    # Get models
    tokenizer, model, image_processor, context_len = get_model(args)
    #  model = model.to(torch.float16)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config).float()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    # model.print_trainable_parameters()

    train_dataset = ReGraPDataset_Mixture(
        data_root=args.data_root,
        set_name=args.set_name,
        tokenizer=tokenizer,
        config=model.config,
        image_processor=image_processor,
        device=model.device,
        flip_p=args.flip_p,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=1
    )

    test_dataset = ReGraPDataset(
        data_root=args.data_root,
        set_name=args.set_name,
        train_image_paths=train_dataset.images_path,
        tokenizer=tokenizer,
        config=model.config,
        image_processor=image_processor,
        device=model.device,
        set='test',
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4
    )

    print('sks is: ', args.set_name)
    print('Number of training samples:', len(train_dataset))

    if args.prefix_token > 0:
        placeholder_tokens = [f'<{args.set_name}>']

        n = args.prefix_token  # n relation tokens
        relation_tokens = [f'<relation_{i}>' for i in range(1, n + 1)]
        entity_tokens = [f'<entity_{i}>' for i in range(1, n + 2)]  # n+1
        placeholder_tokens += entity_tokens
        placeholder_tokens += relation_tokens

        pair_prompts = [
            f"{entity_tokens[i]} and {entity_tokens[i + 1]} share {relation_tokens[i]}"
            for i in range(n)
        ]
        system_prompt = ". ".join(pair_prompts) + "."
        print("system prompt will add:", system_prompt)

    else:
        placeholder_tokens = [f'<{args.set_name}>']
        system_prompt = f"{placeholder_tokens[0]}"
        print('system prompt will add:', system_prompt)
    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    model.resize_token_embeddings(len(tokenizer))

    token_embeds = model.get_input_embeddings().weight.data
    orig_embeds_params = model.get_input_embeddings().weight.data.clone()
    orig_lm_params = model.lm_head.weight.data.clone()

    trainable_params = model.parameters()

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=1e-5,
        betas=(0.9, 0.999),
        weight_decay=0.1,
        eps=1e-08,
        fused=True
    )

    scaler = torch.cuda.amp.GradScaler()
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch * len(train_dataloader), eta_min=1e-7)
    model.train()
    best_acc = 0
    for epoch in tqdm(range(0, args.epoch)):
        epoch_loss = 5

        for step, batch in enumerate(tqdm(train_dataloader)):

            optimizer.zero_grad()
            if args.user_prompt:
                prompt = [
                    get_query(args, system_prompt + ' ' + x, model=model, sks_system_prompt=None).conv.get_prompt()
                    for x in batch['query']]
            else:
                prompt = [get_query(args, x, model=model, sks_system_prompt=system_prompt).conv.get_prompt() for x in
                          batch['query']]

            prompt = [x + ' ' + y for x, y in zip(prompt, batch['answer'])]

            if not batch['has_image']:
                prompt = [x.replace('<image>\n', '') for x in prompt]
            input_ids, labels = tokenizer_image_token_batch(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt",
                                                            return_labels=True)
            input_ids = input_ids.cuda()
            labels = labels.cuda()

            with torch.cuda.amp.autocast(enabled=False, dtype=torch.float16):
                if not batch['has_image']:
                    outputs = model(input_ids, labels=labels)
                else:
                    outputs = model(input_ids, images=batch['images'][0], labels=labels,
                                    image_sizes=batch['image_sizes'])
            loss = outputs.loss
            epoch_loss = loss.item()
            print('Loss: ', epoch_loss)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
            index_no_updates[placeholder_token_ids] = False
            with torch.no_grad():
                model.get_input_embeddings().weight[
                    index_no_updates
                ] = orig_embeds_params[index_no_updates]
                model.lm_head.weight[index_no_updates] = orig_lm_params[index_no_updates]
            # torch.cuda.empty_cache()

            writer.add_scalar('Loss/Train', loss.item(), epoch * len(train_dataloader) + step)
            writer.add_scalar('Loss/Token-Norm',
                              model.get_input_embeddings().weight[placeholder_token_ids].norm().item(),
                              epoch * len(train_dataloader) + step)
            writer.add_scalar('Loss/index_no_updates-Norm',
                              model.get_input_embeddings().weight[index_no_updates].norm().item(),
                              epoch * len(train_dataloader) + step)
            writer.add_scalar('Loss/lm-head-norm', model.lm_head.weight[placeholder_token_ids].norm().item(),
                              epoch * len(train_dataloader) + step)
            writer.add_scalar('Loss/index_no_updates-lm-head', model.lm_head.weight[index_no_updates].norm().item(),
                              epoch * len(train_dataloader) + step)

        # scheduler.step()

        print('Epoch: ', epoch, 'Loss: ', epoch_loss)
        with open(f"./checkpoints/{args.set_name}/loss.txt", "a") as f:
            f.write(f'{epoch}: {epoch_loss}\n')

        if epoch % args.log_every == 0:
            print('Save model at: ', save_location)
            save_path_token = os.path.join(save_location, f'{epoch}-token.pt')
            save_path_lmhead = os.path.join(save_location, f'{epoch}-lmhead.pt')
            save_path_lora = os.path.join(save_location, f'{epoch}-lora')
            torch.save(model.get_input_embeddings().weight.data[placeholder_token_ids], save_path_token)
            torch.save(model.lm_head.weight.data[placeholder_token_ids], save_path_lmhead)
            model.save_pretrained(save_path_lora)
