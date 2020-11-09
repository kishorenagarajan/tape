def test_ckm():
    import numpy as np
    import pandas as pd
    import torch
    from tape import ProteinBertForMultiLabelClassification, ProteinBertConfig, TAPETokenizer  # type: ignore
    from tape.registry import Registry

    model = Registry.get_task_model(model_name="transformer", task_name="protein_domain", config_file="../results/protein_domain_transformer_20-11-04-21-37-19_057939/config.json")
    tokenizer = TAPETokenizer(vocab='iupac')

    sequence = 'GCTVEDRCLIGMGAILLNGCVIGSGSLVAAGALITQ'
    token_ids = torch.tensor([tokenizer.encode(sequence)])
    output = model(token_ids)
    logits = output[0]
    sequence_output = output[1]  # noqa
    pooled_output = output[2]  # noqa

    probs = torch.nn.functional.sigmoid(logits)

    for term in ["probs", "sequence_output", "pooled_output"]:
        print(f"{term}: \n    {eval(term)}")

    probs_np = probs.detach().numpy()
    probs_df = pd.DataFrame(probs_np)
    probs_df.to_csv("../probs.csv")

test_ckm()