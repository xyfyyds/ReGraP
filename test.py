# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from peft import PeftModel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from llava.eval.my_llava import *
from llava.mm_utils import (get_model_name_from_path, tokenizer_image_token_batch)

IMAGE_TOKEN_INDEX = -200


def get_train_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.6-vicuna-13b")
    parser.add_argument("--model_path", type=str, default="./liuhaotian/llava-v1.6-vicuna-7b")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--conv_mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1000)

    parser.add_argument("--data_root", type=str, default='/data')
    parser.add_argument("--set_name", type=str, default='anime_cup')
    parser.add_argument("--prefix_token", type=int, default=4)
    parser.add_argument("--flip_p", type=float, default=0.5)
    parser.add_argument("--user_prompt", default=False, action='store_true')

    parser.add_argument("--tensorboard_path", type=str, default='./runs/')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoints/')
    parser.add_argument("--exp_name", type=str, default='./debug/')
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=20)
    train_args = parser.parse_args()
    return train_args


if __name__ == "__main__":

    args = get_train_args()

    writer = SummaryWriter(os.path.join(args.tensorboard_path, args.sks_name, args.exp_name))
    save_location = os.path.join(args.checkpoint_path, args.sks_name, args.exp_name)
    os.makedirs(save_location, exist_ok=True)
    args.model_name = get_model_name_from_path(args.model_path)

    # Get models
    tokenizer, model, image_processor, context_len = get_model(args)

    placeholder_tokens = [f'<{args.set_name}>']

    n = args.prefix_token  # n relation tokens
    relation_tokens = [f'<relation_{i}>' for i in range(1, n + 1)]
    entity_tokens = [f'<entity_{i}>' for i in range(1, n + 2)]  # n+1
    placeholder_tokens += entity_tokens
    placeholder_tokens += relation_tokens

    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
    model.resize_token_embeddings(len(tokenizer))

    lora_model_path = f'{args.checkpoint_path}/{args.sks_name}/feature/4-lora'
    model = PeftModel.from_pretrained(model, lora_model_path)
    model = model.merge_and_unload()  # Merge LoRA into base model

    print("✅ LoRA successfully merged into the base model!")

    sks_token = torch.load(f'{args.checkpoint_path}/anime_cup/feature/4-token.pt').detach()
    lm_head = torch.load(f'{args.checkpoint_path}/anime_cup/feature/4-lmhead.pt').detach()
    model.get_input_embeddings().weight.requires_grad = False
    model.lm_head.weight.requires_grad = False
    model.get_input_embeddings().weight[placeholder_token_ids] = sks_token.to(model.device, dtype=model.dtype)
    model.lm_head.weight[placeholder_token_ids] = lm_head.detach().to(model.lm_head.weight.device, dtype=model.dtype)
    print('New tokens are loaded into: ', placeholder_token_ids)

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

    # breakpoint()

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
    print(test_dataset.image_paths)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4
    )

    print('set is: ', args.set_name)
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
        placeholder_tokens = [f'<{args.sks_name}>']
        system_prompt = f"{placeholder_tokens[0]}"
        print('system prompt will add:', system_prompt)
    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    model.eval()

    best_acc = 0
    for epoch in tqdm(range(0, args.epoch)):

        with torch.no_grad():
            print('Test')
            list_pred = []
            list_gt = []
            for j, batch in enumerate(tqdm(test_dataloader)):

                if args.user_prompt:
                    prompt = [
                        get_query(args, system_prompt + ' ' + x, model=model, sks_system_prompt=None).conv.get_prompt() for
                        x in batch['query']]
                    print('Prompt:', prompt)
                else:
                    prompt = [get_query(args, x, model=model, sks_system_prompt=system_prompt).conv.get_prompt() for x in
                              batch['query']]
                input_ids, labels = tokenizer_image_token_batch(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                                                return_tensors="pt", return_labels=True)
                outputs = model.generate(input_ids.cuda(), images=batch['images'][0].cuda(),
                                         image_sizes=batch['image_sizes'], max_new_tokens=args.max_new_tokens)
                answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                print("Prompt:", prompt)
                print('Prediction:', answer)
                # print('GT:', batch['answer'][0])

            with open(f"./example_training_data/{args.sks_name}/QA.json", "r") as f:
                qa_data = json.load(f)

            multiple_choice_questions = qa_data["multiple_choice"]

            results = []
            correct_mc = 0
            total_mc = 0
            correct_tf = 0
            total_tf = 0
            i = 0

            for j, batch in enumerate(tqdm(test_dataloader)):
                if i >= 1:
                    break
                i += 1
                image_tensor = batch["images"][0].cuda()
                image_sizes = batch["image_sizes"]

                image_results = {"image_id": j, "answers": []}

                for q in multiple_choice_questions:
                    question_text = "Answer the multiple choice question and select one letter. " + q[
                        "question"] + " Choices are: " + str(
                        q["options"]) + " Answer the question using a single word or phrase."
                    prompt = [get_query(args, question_text, model=model).conv.get_prompt()]
                    print("Prompt:", prompt)
                    input_ids, labels = tokenizer_image_token_batch(
                        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt", return_labels=True
                    )
                    outputs = model.generate(
                        input_ids.cuda(), images=image_tensor, image_sizes=image_sizes,
                        max_new_tokens=args.max_new_tokens
                    )
                    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                    print("Answer:", answer)
                    print("Correct Answer:", q["correct_answer"])

                    if answer.strip() == q["correct_answer"]:
                        correct_mc += 1
                    total_mc += 1

                    image_results["answers"].append({"question": q["question"], "answer": answer})
                print("Correct MC:", correct_mc)
                print("Total MC:", total_mc)
                print("Accuracy:", correct_mc / total_mc)
