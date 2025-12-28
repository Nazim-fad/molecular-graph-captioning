import torch
from tqdm import tqdm
from sacrebleu import corpus_bleu
from bert_score import score as bertscore

@torch.no_grad()
def validate_generation(model, tokenizer, val_loader, device, max_new_tokens=128):
    """
    Runs generation on the validation set and computes BLEU-4 and BERTScore.
    """
    model.eval()

    generated_texts = []
    reference_texts = []

    for batch in tqdm(val_loader, desc="Validating Generation"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        # Generation with greedy decoding 
        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,  
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id
        )

        # Slice off the prompt [ prompt | generated ]
        new_tokens = gen_ids[0][input_ids.shape[1]:]
        gen_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Decode reference (ground truth)
        valid_label_ids = labels[0][labels[0] != -100]
        ref_text = tokenizer.decode(valid_label_ids, skip_special_tokens=True).strip()

        generated_texts.append(gen_text)
        reference_texts.append(ref_text)

    # BLEU-4
    try:
        # sacrebleu takes list of hypotheses and list of list of references
        bleu = corpus_bleu(generated_texts, [reference_texts]).score
    except Exception:
        bleu = 0.0

    # BERTScore
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
        print(f"Warning: BERTScore failed ({e}), setting to 0")
        bert_f1 = 0.0

    return {
        "BLEU-4": bleu,
        "BERTScore-F1": bert_f1
    }