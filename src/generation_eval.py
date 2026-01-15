import torch
from tqdm import tqdm
from sacrebleu import corpus_bleu
from bert_score import score as bertscore

@torch.no_grad()
def validate_generation(model, tokenizer, val_loader, device, max_new_tokens=220):
    """
    Runs generation on the validation set and computes BLEU-4 and BERTScore.
    Includes 'Head + Tail' truncation to preserve instructions for long prompts.
    """
    model.eval()

    INSTRUCTION_TOKEN_LEN = 55
    MAX_PROMPT_LEN = 1024 - max_new_tokens - 5

    generated_texts = []
    reference_texts = []

    for batch in tqdm(val_loader, desc="Validating Generation"):
        input_ids_batch = batch["input_ids"].to(device)
        labels_batch = batch["labels"].to(device)

        batch_size = input_ids_batch.shape[0]

        for i in range(batch_size):
            labels = labels_batch[i]
            input_ids = input_ids_batch[i]

            # Identify prompt tokens
            prompt_mask = (labels == -100)

            if not torch.any(prompt_mask):
                prompt_ids = input_ids
            else:
                prompt_ids = input_ids[prompt_mask]

            # Truncation (Head + Tail) on the Prompt Only
            if len(prompt_ids) > MAX_PROMPT_LEN:
                head_len = MAX_PROMPT_LEN - INSTRUCTION_TOKEN_LEN
                
                # Slice Head (Start) and Tail (Instruction)
                prompt_head = prompt_ids[:head_len]
                prompt_tail = prompt_ids[-INSTRUCTION_TOKEN_LEN:]
                
                # Recombine
                prompt_ids = torch.cat([prompt_head, prompt_tail])

            # Batch Dimension
            curr_input_ids = prompt_ids.unsqueeze(0) 
            curr_attention_mask = torch.ones_like(curr_input_ids).to(device)

            # Generate
            try:
                gen_ids = model.generate(
                    input_ids=curr_input_ids,       
                    attention_mask=curr_attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False, 
                    num_beams=1,
                    pad_token_id=tokenizer.eos_token_id
                )
            except Exception as e:
                print(f"Gen Error: {e}")
                continue

            # Slice off the prompt [ prompt | generated ]
            new_tokens = gen_ids[0][curr_input_ids.shape[1]:]
            gen_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            # Decode reference (Target)
            valid_label_ids = labels[labels != -100]
            ref_text = tokenizer.decode(valid_label_ids, skip_special_tokens=True).strip()

            generated_texts.append(gen_text)
            reference_texts.append(ref_text)

    try:
        bleu = corpus_bleu(generated_texts, [reference_texts]).score
    except:
        bleu = 0.0

    try:
        P, R, F1 = bertscore(
            generated_texts, 
            reference_texts, 
            lang="en", 
            model_type="distilbert-base-uncased", 
            verbose=False
        )
        bert_f1 = F1.mean().item()
    except Exception as e:
        print(f"BERTScore error: {e}")
        bert_f1 = 0.0

    return {
        "BLEU-4": bleu,
        "BERTScore-F1": bert_f1
    }


@torch.no_grad()
def validate_generation_molt5(model, val_loader, device):
    model.eval()

    generated_texts = []
    reference_texts = []

    for batch in tqdm(val_loader, desc="Validating MolT5"):
        batch_graph = batch["batch_graph"].to(device)
        labels = batch["labels"]

        gens = model.generate(batch_graph)

        for gen_text, label_ids in zip(gens, labels):
            valid_ids = label_ids[label_ids != -100]
            ref_text = model.tokenizer.decode(
                valid_ids, skip_special_tokens=True
            )

            generated_texts.append(gen_text.strip())
            reference_texts.append(ref_text.strip())

    bleu = corpus_bleu(generated_texts, [reference_texts]).score
    _, _, F1 = bertscore(generated_texts, reference_texts, lang="en")

    return {
        "BLEU-4": bleu,
        "BERTScore-F1": F1.mean().item()
    }
