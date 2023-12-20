import torch as th

def embsimScore(model, emb_model, idf_dict, hyp_emb, model_kwargs, device):
        
    ref_idf = th.tensor([[idf_dict[ids.item()] for ids in src_ids]for src_ids in model_kwargs['src_input_ids']])
    ref_idf = ref_idf.div(ref_idf.sum(dim=1, keepdim=True))
    
    device = next(model.parameters()).device
    
    ref_emb = emb_model(model_kwargs['src_input_ids'].to(device))    
    ref_emb = ref_emb.div(th.norm(ref_emb, dim=-1).unsqueeze(-1))
    
    logits = model.module.get_logits(hyp_emb)
    cands = th.topk(logits, k=1, dim=-1)
    sample_id_list = cands.indices
    hyp_token = sample_id_list.squeeze()
    
    hyp_idf = th.tensor([[idf_dict[ids.item()] for ids in hyp_ids]for hyp_ids in hyp_token])
    hyp_idf = hyp_idf.div(hyp_idf.sum(dim=1, keepdim=True))
    
    hyp_emb = hyp_emb.div(th.norm(hyp_emb, dim=-1).unsqueeze(-1))

    sim = th.bmm(hyp_emb, ref_emb.transpose(1, 2))
    
    word_precision = sim.max(dim=2)[0]
    word_recall = sim.max(dim=1)[0]

    precision_scale = hyp_idf.to(word_precision.device)
    recall_scale = ref_idf.to(word_recall.device)

    P = (word_precision * precision_scale).sum(dim=1)
    R = (word_recall * recall_scale).sum(dim=1)
    F = 2 * P * R / (P + R)

    loss = th.ones(1).to(device) - F.mean()
    
    return loss

def langevin_fn_BertScore(model, emb_model, idf_dict, sample,  model_kwargs, mean, sigma):
    
    device = model_kwargs['src_input_ids'].device
    input_embs_param = th.nn.Parameter(sample)
    coef=0.0001
    
    with th.enable_grad():
        optimizer = th.optim.Adagrad([input_embs_param], lr=0.0001)
        optimizer.zero_grad()

        score_loss = embsimScore(model, emb_model, idf_dict, input_embs_param, model_kwargs, device)

        if sigma.mean() == 0:
            logp_term = coef * ((mean - input_embs_param) ** 2 / 1.).mean(dim=0).sum()
        else:
            logp_term = coef * ((mean - input_embs_param)**2 / sigma).mean(dim=0).sum()


        loss = score_loss + logp_term.item()

        loss.mean().backward()
        optimizer.step()

        epsilon = th.randn_like(input_embs_param.data)
        input_embs_param = th.nn.Parameter((input_embs_param.data + 0.0*sigma.mean().item() * epsilon).detach())
        
    return input_embs_param.data