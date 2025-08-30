//
// Created by Jonathan Middleton on 8/29/25.
//
// SPDX-License-Identifier: MIT
#include "llama.h"
#include <vector>
#include <deque>
#include <algorithm>
#include <limits>
#include <cstring>

struct smpl_rbudget_ctx {
    std::vector<llama_token> open;
    std::vector<llama_token> close;
    std::deque<llama_token>  window;
    std::deque<llama_token>  forceq;

    uint32_t   budget = 0;
    uint32_t   used   = 0;
    bool       inside = false;
    bool       hard   = true;
    float      close_bias = 0.0f;

    int        max_win = 0;

    static bool ends_with(const std::deque<llama_token> &w, const std::vector<llama_token> &pat) {
        if ((int)w.size() < (int)pat.size()) { return false;
}
        auto it = w.end();
        for (int i = (int)pat.size()-1; i >= 0; --i) {
            --it;
            if (*it != pat[i]) { return false;
}
        }
        return true;
    }

    void push_tok(llama_token t) {
        window.push_back(t);
        if ((int)window.size() > max_win) { window.pop_front();
}

        if (!inside) {
            if (!open.empty() && ends_with(window, open)) {
                inside = true;
                used   = 0;
                forceq.clear();
            }
        } else {
            if (!close.empty() && ends_with(window, close)) {
                inside = false;
                used   = 0;
                forceq.clear();
            } else {
                used++;
            }
        }
    }

    void begin_force_close() {
        if (forceq.empty()) {
            for (auto t : close) { forceq.push_back(t);
}
        }
    }
};

static const char * smpl_rbudget_name(const struct llama_sampler * /*unused*/) { return "reasoning_budget"; }

static void smpl_rbudget_accept(struct llama_sampler * smpl, llama_token tok) {
    auto * s = (smpl_rbudget_ctx *) smpl->ctx;
    if (!s->forceq.empty() && tok == s->forceq.front()) { s->forceq.pop_front();
}
    s->push_tok(tok);
}


static void smpl_rbudget_reset(struct llama_sampler * smpl) {
    auto * s = (smpl_rbudget_ctx *) smpl->ctx;
    s->window.clear();
    s->forceq.clear();
    s->used   = 0;
    s->inside = false;
}

static void clamp_to_token(llama_token_data_array * cur_p, llama_token want) {
    // ensure 'want' is present; if not, overwrite the last slot
    int64_t idx = -1;
    for (size_t i = 0; i < cur_p->size; ++i) {
        if (cur_p->data[i].id == want) { idx = (int64_t)i; break; }
    }
    if (idx == -1 && cur_p->size > 0) {
        cur_p->data[cur_p->size - 1].id = want;
        idx = (int64_t)cur_p->size - 1;
    }
    const float INF = std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < cur_p->size; ++i) {
        cur_p->data[i].logit = (i == (size_t)idx) ? 1e9f : -INF;
    }
    cur_p->selected = idx;
    cur_p->sorted   = false;
}

static void bias_token(llama_token_data_array * cur_p, llama_token want, float bias) {
    for (size_t i = 0; i < cur_p->size; ++i) {
        if (cur_p->data[i].id == want) {
            cur_p->data[i].logit += bias;
            break;
        }
    }
}

static void smpl_rbudget_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * s = (smpl_rbudget_ctx *) smpl->ctx;

    if (!s->inside) { return;
}
    if (s->budget == 0) {  return;
}

    if (s->used < s->budget) { return;
}

    // budget exhausted
    if (s->hard) {
        if (s->forceq.empty()) { s->begin_force_close();
}
        if (!s->forceq.empty()) {
            clamp_to_token(cur_p, s->forceq.front());
        }
    } else {
        // soft mode: bias towards the first close token
        if (!s->close.empty()) {
            bias_token(cur_p, s->close.front(), s->close_bias);
        }
    }
}

static struct llama_sampler * smpl_rbudget_clone(const struct llama_sampler * smpl) {
    auto * old = (smpl_rbudget_ctx *) smpl->ctx;
    auto * neu = new smpl_rbudget_ctx(*old);
    neu->window.clear();
    neu->forceq.clear();
    neu->used   = 0;
    neu->inside = false;
    return llama_sampler_init(smpl->iface, neu);
}

static void smpl_rbudget_free(struct llama_sampler * smpl) {
    delete (smpl_rbudget_ctx *) smpl->ctx;
}

static const struct llama_sampler_i SMPL_RB_I = {
    /*.name   =*/ smpl_rbudget_name,
    /*.accept =*/ smpl_rbudget_accept,
    /*.apply  =*/ smpl_rbudget_apply,
    /*.reset  =*/ smpl_rbudget_reset,
    /*.clone  =*/ smpl_rbudget_clone,
    /*.free   =*/ smpl_rbudget_free,
};

// src/llama-sampling-reasoning-budget.cpp
extern "C" LLAMA_API struct llama_sampler *
llama_sampler_init_reasoning_budget(const llama_vocab * vocab,
                                    uint32_t             budget_tokens,
                                    const char *         open_tag,
                                    const char *         close_tag,
                                    float                close_bias,
                                    bool                 hard_enforce) {
    auto * s = new smpl_rbudget_ctx;
    s->budget     = budget_tokens;
    s->hard       = hard_enforce;
    s->close_bias = close_bias;

    auto tok = [&](const char *str, std::vector<llama_token> &out) {
        out.clear();
        if (!str || !*str) { return;
}
        const auto len  = (int32_t) std::strlen(str);
        const int32_t r1   = llama_tokenize(vocab, str, len, /*tokens*/nullptr, /*n_max*/0,
                                            /*add_special*/false, /*parse_special*/true);
        const int32_t need = r1 >= 0 ? r1 : -r1;       // negative => required size
        if (need <= 0) { return;
}
        out.resize(need);
        const int32_t r2 = llama_tokenize(vocab, str, len, out.data(), need,
                                          /*add_special*/false, /*parse_special*/true);
        // r2 should be == need; if not, clamp to the absolute value
        if (r2 != need) out.resize(r2 >= 0 ? r2 : -r2);
    };

    tok(open_tag,  s->open);
    tok(close_tag, s->close);
    s->max_win = std::max({(int)s->open.size(), (int)s->close.size(), 8});

    return llama_sampler_init(&SMPL_RB_I, s);
}
