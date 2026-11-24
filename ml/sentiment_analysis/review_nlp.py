"""Portuguese review intelligence: complaint taxonomy, aspect-based sentiment, severity.

Everything is rule/lexicon based so it is free, local, transparent and explainable in a viva.
Text is lower-cased and accent-stripped before matching, so patterns below have no accents.
Taxonomy follows the project PDF (10 classes). A review can belong to several complaint categories.
Validation against a hand-labelled sample is done in ml/evaluation/evaluate_nlp.py.
"""
import re
import unicodedata

COMPLAINT_CATEGORIES = [
    "late_delivery", "damaged_product", "wrong_product", "missing_product", "product_quality",
    "seller_issue", "packaging", "shipping", "payment", "other",
]

# phrases that look like delay words but are praise -> removed before matching
_PRAISE_PHRASES = re.compile(
    r"\b(dentro do prazo|no prazo|antes do prazo|antes da data|antes do previsto|antes do esperado|chegou antes|"
    r"muito antes|bem antes|nao (houve|teve) atraso|sem atraso|nao atrasou|nao demorou|nao tive problema)\b")

PATTERNS = {
    "late_delivery": r"\batras|demor|nao (chegou|recebi|foi entregue|entregaram|veio|recebemos|chegaram)|"
                     r"ainda nao (chegou|recebi|veio|foi|recebemos|chegaram)|nao foi entregue|nao entregue|nao entregaram|"
                     r"(aguardando|aguardo|no aguardo|agurdo|esperando)(?!\s*(a |o |um |uma |seu |sua |por )?(retorno|resposta|contato|solucao|troca|devolucao|e mail|email|reembolso|estorno|vendedor|loja|empresa|ressarcimento|posicionamento|alguma))|ate (hoje|agora|o momento)|nunca chegou|"
                     r"passou d[oa] (prazo|data)|data prevista|previsao de entrega|fora do prazo|alem do prazo|"
                     r"prazo (estourad|vencid)|nao recebi|sem receber|nao chegaram|ainda nao|extrapol",
    "damaged_product": r"quebrad|quebrou|rachad|amassad|danificad|avaria|estragad|trincad|arranhad|riscad|rasgad|"
                       r"furad|vazand|vazou|vazad|molhad|manchad|amassou|partido|quebr[ao]u|chegou (todo )?(quebr|danific|estrag)|"
                       r"desfiad|descascand|descascou|rasgou|com marcas|machucad|empenad",
    "wrong_product": r"produto (errado|diferente|trocado)|errad[oa]s?\b|diferente d[oa]|diferente do (anunciado|pedido|solicitado)|"
                     r"nao (era|e) (o|a) (que|produto)|(enviaram|mandaram|recebi|veio|vieram|entregaram|enviado|enviou|mandou) "
                     r"(outro|outra|errad|diferente|trocad)|cor (diferente|errada)|tamanho (errado|diferente)|"
                     r"nao corresponde|ao inves d[eo]|outro produto|outra cor|nao e original|"
                     r"nao (foi|era) o que (pedi|comprei|solicitei)|nao pedi|nao e o produto",
    "missing_product": r"faltou|faltando|faltam|falta d[eoa]s?\b|incomplet|so recebi (um|uma|1|dois|duas|2|parte)|"
                       r"(apenas|somente|so) (um|uma|1|dois|duas|2) (dos|das|item|produto|unidade|pecas?)|"
                       r"recebi (apenas|somente|so|parte)|veio (so|apenas|somente)|nao veio (o|a|os|as|completo|tudo)|"
                       r"sem (o|a|os|as|um|uma) (acessorio|peca|parafuso|manual|cabo|carregador|controle|tampa|fonte)|"
                       r"kit incompleto|metade (do|da|dos|das)|nao recebi (todos|todo|o restante|a outra|o outro|os demais)|"
                       r"(1|um|uma|dois|duas|2) (dos|das) (\d|dois|duas|tres)|restante (do|dos) (pedido|produtos|itens)|"
                       r"outro item|outra unidade|faltaram|pedido incompleto|entrega parcial|so veio",
    "product_quality": r"defeito|defeituos|nao funciona|nao funcionou|nao funcionam|parou de funcionar|"
                       r"(baixa|pessima|ma|fraca|duvidosa|horrivel) qualidade|qualidade (ruim|pessima|duvidosa|inferior|baixa|"
                       r"fraca|horrivel|deixa a desejar)|nao tem qualidade|sem qualidade|produto (ruim|pessimo|fraco|inferior|"
                       r"horrivel|falsificado|pirata)|falsific|pirata|nao liga|nao carrega|nao acende|nao abre|nao vale|"
                       r"queimou|esquent|travando|trava\b|descarrega|desbotad|enferruj|nao presta|lixo|inferior|decepcion|"
                       r"fragil|frag[i]l|nao serve|nao encaixa|nao cabe|nao vedam|barulho|cheiro|material (ruim|fraco|inferior)|"
                       r"propaganda enganosa|nao (e|eh) (de )?(boa|bom)|descolou|soltou|desmont|nao gostei|mal feit|nao recomendo o produto",
    "seller_issue": r"(nao|nunca) (consegui|tive|obtive|houve|tem|ha|recebi) (nenhum )?(contato|resposta|retorno|atendimento|"
                    r"satisfacao|solucao)|sem (resposta|retorno|atendimento|contato|solucao)|ninguem (responde|resolve|respondeu|"
                    r"retorna|atende)|nao (respond|resolv|retorn|atend)|descaso|desrespeito|falta de respeito|"
                    r"falta de (atencao|compromisso|profissionalismo|comunicacao)|anuncio enganoso|enganos|enganad|"
                    r"reclame aqui|reclameaqui|procon|golpe|fraude|estelion|nota fiscal|\bnf\b|\bnfe\b|"
                    r"pessimo atendimento|atendimento (pessimo|ruim|horrivel|precario)|loja (pessima|ruim|nao confiavel)|"
                    r"vendedor (pessimo|ruim|nao)|(solicito|solicitei|quero|preciso|exijo|aguardo) (a )?(troca|devolucao|"
                    r"cancelamento|reembolso|retorno|solucao)|cancelaram|cancelamento|sem previsao|nao se pronuncia|"
                    r"nao me deram|nao deram|nao tive solucao|devolucao|devolver|troca\b|trocar",
    "packaging": r"embalag|caixa (amassad|rasgad|aberta|violad|danific|molhad)|lacre|violad|sem protecao|mal embalad|"
                 r"protecao|papelao|plastico bolha|veio aberto|veio aberta|sem caixa|sem embalagem",
    "shipping": r"correios|transportadora|extravi|postad|rastre|codigo de rastr|agencia|centro de distribuicao|"
                r"devolvid|retornou ao remetente|entregue (para|a|em) (outra|outro|terceir|vizinh)|entregue em outro|"
                r"endereco (errado|incorreto|diferente)|entregador|motoboy|frete (caro|alto|abusivo)|paguei (o )?frete|"
                r"greve|parad[ao] (em|no|na|nos)|em transito|tentativa de entrega|ausente|nao localizad|entregue (no|na) "
                r"(endereco|portaria)|consta (como )?entregue|consta que foi entregue|foi entregue (mas|porem)|encomenda|"
                r"despach|postou|postagem|sedex|\bpac\b|logistic|mercadoria parada",
    "payment": r"cobranc|cobrad|cartao|boleto|estorno|reembols|parcela|pagamento|debitad|dinheiro de volta|"
               r"valor (cobrado|total)|duplicidade|cobraram|credito|debito|desconto nao|dinheiro de volta|pix|"
               r"pagar (duas|2|em duplicidade)|cobrou",
}
_COMPILED = {k: re.compile(v) for k, v in PATTERNS.items()}

# ---------------------------------------------------------------- aspects
ASPECTS = {
    "delivery": r"entreg|prazo|chegou|chegar|chegaram|atras|demor|frete|correio|transportadora|rapid|recebi|recebe|"
                r"antes d|aguard|postad|rastre|despach|envio|enviad|enviou|pontual|agil",
    "product": r"produto|qualidade|funcion|defeito|material|tamanho|cor\b|cores|bonito|lindo|linda|quebrad|original|"
               r"fragil|cheiro|aparelho|item\b|itens|mercadoria|artigo|modelo|durabil|acabamento|cabe",
    "seller_service": r"vendedor|vendedora|loja|atendimento|contato|resposta|responder|respondeu|empresa|sac\b|retorno|"
                      r"anuncio|nota fiscal|devolucao|troca|reembolso|suporte|parceiro|lojista",
    "packaging": r"embalag|caixa|lacre|papelao|protecao|embalado",
    "price": r"preco|valor|barato|caro\b|custo|beneficio|promocao|desconto",
    "payment": r"cobranc|pagamento|cartao|boleto|estorno|reembols|parcela",
}
_ASPECTS_C = {k: re.compile(v) for k, v in ASPECTS.items()}

POS_WORDS = re.compile(
    r"\b(otim[oa]s?|excelente|excelentes|bom|boa|boas|bons|gostei|adorei|amei|perfeit[oa]s?|maravilh\w+|lind[oa]s?|"
    r"recomendo|satisfeit[oa]s?|rapid[oa]s?|rapidez|certinho|supera\w*|top|parabens|show|bonito|bonita|eficiente|pontual|"
    r"agil|veloz|conforme|sem problemas?|tudo certo|tudo ok|ok|chegou antes|antes do prazo|no prazo|dentro do prazo|"
    r"surpreend\w+|cumpr\w+|ideal|fantastic\w+|nota 10|melhor|feliz|obrigad[oa]|bem embalad\w+|barato|vale a pena|"
    r"qualidade|confiavel|tranquil\w+|facil|resistente|bem feito)\b")
NEG_WORDS = re.compile(
    r"\b(pessim[oa]s?|ruim|ruins|horrivel|horriveis|lixo|decepc\w+|nao gostei|odiei|defeito|defeituos[oa]|errad[oa]s?|"
    r"quebrad[oa]s?|atras\w+|demor\w+|faltou|faltando|faltaram|incomplet\w+|fraco|fraca|inferior|insatisf\w+|reclam\w+|"
    r"absurdo|descaso|desrespeito|problemas?|danificad[oa]s?|nao recomendo|frustr\w+|enganos[oa]|enganad[oa]|falsific\w+|"
    r"decepcion\w+|vergonha|golpe|fraude|ridicul\w+|pior|amassad[oa]s?|rachad[oa]s?|avaria|arrependi\w+|"
    r"nunca mais|dor de cabeca|caro|abusivo|precario|nao funciona|nao funcionou|nao chegou|nao recebi|nao veio|nao liga|"
    r"nao entregue|nao entregaram|nao foi entregue|ainda nao|nao carrega|nao serve|nao cabe|aguardando|esperando|"
    r"sem resposta|nao respond\w+|nao resolv\w+|diferente)\b")
_NEG_PREFIX = re.compile(r"\b(nao|nunca|nem|jamais|sem)\s+(\w+\s+)?$")
_SPLIT = re.compile(r"[.!?;\n]+|\bmas\b|\bporem\b|\bcontudo\b|\bentretanto\b|\btodavia\b|\bso que\b|\bno entanto\b")

_SEVERE = re.compile(r"reclame aqui|reclameaqui|procon|golpe|fraude|estelion|absurdo|descaso|desrespeito|nunca mais|"
                     r"pessim|horrivel|vergonha|ridicul|enganos|enganad|processo|justica|advogado|nota zero|nao recomendo|"
                     r"lixo|pior|revoltad|indignad|roubo|roubad")
_HIGH_CATS = {"late_delivery", "missing_product", "wrong_product", "damaged_product", "payment"}


def normalize(text):
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)          # soooo -> soo
    t = re.sub(r"\bn\b", "nao", t)                   # common abbreviation
    t = re.sub(r"\bnao\b", "nao", t)
    t = re.sub(r"\bnn\b|\bnaum\b|\bnum\b(?= (foi|chegou|recebi|veio))", "nao", t)
    t = re.sub(r"\bvc\b", "voce", t)
    t = re.sub(r"[^a-z0-9\s.!?;]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def complaint_categories(norm_text):
    """Return list of complaint categories found. 'other' is returned only if nothing else matched."""
    t = _PRAISE_PHRASES.sub(" ", norm_text)
    found = [c for c in COMPLAINT_CATEGORIES[:-1] if _COMPILED[c].search(t)]
    return found or ["other"]


def _sentence_polarity(sent):
    """Lexicon polarity of one sentence with simple negation handling. Returns int score."""
    score = 0
    for m in NEG_WORDS.finditer(sent):
        score -= 1
    for m in POS_WORDS.finditer(sent):
        prefix = sent[:m.start()]
        if _NEG_PREFIX.search(prefix[-25:]):
            score -= 1                          # "nao gostei" style flips praise to negative
        else:
            score += 1
    # negated negative ("sem problemas", "nao tive problema") -> positive
    for m in re.finditer(r"\b(sem|nao tive|nao teve|nenhum|nao houve)\s+(problemas?|defeito|atraso)", sent):
        score += 2
    return score


def sentence_sentiment(norm_text):
    return [(s.strip(), _sentence_polarity(s)) for s in _SPLIT.split(norm_text) if s and s.strip()]


def aspect_sentiment(norm_text):
    """Return {aspect: 'positive'|'negative'|'neutral'|None} (None = aspect not mentioned)."""
    out = {a: None for a in ASPECTS}
    sents = sentence_sentiment(norm_text)
    for a, rx in _ASPECTS_C.items():
        net, hit = 0, False
        for s, pol in sents:
            if rx.search(s):
                hit = True
                net += pol
        if hit:
            out[a] = "positive" if net > 0 else "negative" if net < 0 else "neutral"
    return out


def overall_lexicon_sentiment(norm_text):
    net = sum(p for _, p in sentence_sentiment(norm_text))
    return "positive" if net > 0 else "negative" if net < 0 else "neutral"


def severity(norm_text, categories):
    """Rule-based severity: High / Medium / Low (text cues + category weight)."""
    pts = 0
    pts += 2 if _SEVERE.search(norm_text) else 0
    pts += 2 if any(c in _HIGH_CATS for c in categories) else 0
    pts += 1 if any(c in {"product_quality", "seller_issue", "shipping"} for c in categories) else 0
    pts += 1 if len(categories) >= 2 and categories != ["other"] else 0
    pts += 1 if norm_text.count("!") >= 2 or re.search(r"\b[a-z]{3,}\b", norm_text) and len(norm_text.split()) > 25 else 0
    return "High" if pts >= 4 else "Medium" if pts >= 2 else "Low"


def analyse(text):
    """Full analysis of one raw review text -> dict (used by pipeline, evaluation and the Flask app)."""
    n = normalize(text)
    cats = complaint_categories(n)
    return {
        "norm_text": n,
        "complaint_categories": cats,
        "aspects": aspect_sentiment(n),
        "lexicon_sentiment": overall_lexicon_sentiment(n),
        "severity": severity(n, cats),
    }
