import torch
import torch.nn as nn
import torch.nn.functional as F

from embedding import Embedding


class Net(nn.Module):
    def __init__(self, config, vocab):
        super(Net, self).__init__()
        self.embed = Embedding(config, vocab)

        def gen_convs(in_channel, kernel_sizes, output_channels):
            return nn.ModuleList([
                nn.Conv1d(
                    in_channels=in_channel,
                    out_channels=oc,
                    kernel_size=kz,
                    padding=((kz - 1) // 2))
                for kz, oc in zip(kernel_sizes, output_channels)])

        full_size = sum(config.output_channels)

        self.convs_QA = gen_convs(
            config.q_seq_len,
            config.kernel_sizes,
            config.output_channels)
        self.convs_QR = gen_convs(
            config.q_seq_len,
            config.kernel_sizes,
            config.output_channels)
        self.convs_CA = gen_convs(
            config.c_seq_len,
            config.kernel_sizes,
            config.output_channels)
        self.convs_CR = gen_convs(
            config.c_seq_len,
            config.kernel_sizes,
            config.output_channels)
        self.convs_PQ = gen_convs(
            full_size,
            config.kernel_sizes,
            config.output_channels)
        self.convs_PC = gen_convs(
            full_size,
            config.kernel_sizes,
            config.output_channels)
        self.drop_QA = nn.Dropout(config.dropout)
        self.drop_QR = nn.Dropout(config.dropout)
        self.drop_CA = nn.Dropout(config.dropout)
        self.drop_CR = nn.Dropout(config.dropout)
        self.drop_PQ = nn.Dropout(config.dropout)
        self.drop_PC = nn.Dropout(config.dropout)
        self.proj1 = nn.Linear(full_size, full_size)
        self.proj2 = nn.Linear(full_size, 1)

    def forward(self, p, q, cs):
        """
        P: B x N(P) x L(P)
        Q: B x L(Q)
        C: [B x L(Q)] x N(C)
        """
        bs = p.size(0)
        ps = p.size(1)
        pl = p.size(2)
        cl = cs[0].size(1)

        p_flat = p.view(bs, -1)
        p_emb = self.embed(p_flat)
        q_emb = self.embed(q)
        c_embs = [self.embed(c) for c in cs]
        p_norm = p_emb / p_emb.norm(dim=-1)[:, :, None]
        q_norm = q_emb / q_emb.norm(dim=-1)[:, :, None]
        c_norms = [c_emb / c_emb.norm(dim=-1)[:, :, None] for c_emb in c_embs]

        # Calculate the similarity matrices
        mat_pq = torch.matmul(p_norm, q_norm.permute(0, 2, 1))
        mat_pq = mat_pq.reshape(bs * ps, pl, -1).permute(0, 2, 1)
        mat_pcs = []
        for c_norm in c_norms:
            mat_pc = torch.matmul(p_norm, c_norm.permute(0, 2, 1))
            mat_pc = mat_pc.reshape(bs * ps, pl, -1).permute(0, 2, 1)
            mat_pcs.append(mat_pc)

        def conv_op(I, convs):
            return torch.cat([conv(I) for conv in convs], dim=1)
        # Stage 1 attention
        qa = torch.max(conv_op(mat_pq, self.convs_QA), dim=-1)[0].unsqueeze(-1)
        qa = self.drop_QA(F.sigmoid(qa))

        qr = self.drop_QR(F.relu(conv_op(mat_pq, self.convs_QR)))
        qr = torch.max(qr * qa, dim=2)[0].reshape(bs, ps, -1).permute(0, 2, 1)

        cas = [
            torch.max(conv_op(mat_pc, self.convs_CA), dim=-1)[0].unsqueeze(-1)
            for mat_pc in mat_pcs]
        cas = [self.drop_CA(F.sigmoid(ca)) for ca in cas]

        crs = [
            self.drop_CR(F.relu(conv_op(mat_pc, self.convs_CR)))
            for mat_pc in mat_pcs]
        crs = [
            torch.max(cr * ca, dim=2)[0].reshape(bs, ps, -1).permute(0, 2, 1)
            for cr, ca in zip(crs, cas)]

        # Stage 2 attention
        pq = torch.max(conv_op(qr, self.convs_PQ), dim=-1)[0].unsqueeze(-1)
        pq = self.drop_PQ(F.sigmoid(pq))

        pcs = [self.drop_PC(F.relu(conv_op(cr, self.convs_PC))) for cr in crs]
        pcs = torch.stack([torch.max(pq * pc, dim=2)[0] for pc in pcs], dim=1)

        logits = F.softmax(
            self.proj2(F.tanh(self.proj1(pcs))).squeeze(-1),
            dim=-1)

        return logits
