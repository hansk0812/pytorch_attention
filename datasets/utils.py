import string

def return_unicode_hex_within_range(start, num_chars):
    assert isinstance(start, str) and isinstance(num_chars, int)
    start_integer = int(start, 16)
    return ["".join("{:02x}".format(start_integer+x)).lower() for x in range(num_chars)]

def return_tamil_unicode_isalnum():
    ayudha_yezhuthu_stressing_connector = return_unicode_hex_within_range("b82", 2)
    a_to_ooo = return_unicode_hex_within_range("b85", 6)
    ye_to_i = return_unicode_hex_within_range("b8e", 3)
    o_O_ou_ka = return_unicode_hex_within_range("b92", 4)
    nga_sa = return_unicode_hex_within_range("b99", 2)
    ja = return_unicode_hex_within_range("b9c", 1)
    nya_ta = return_unicode_hex_within_range("b9e", 2)
    Na_tha = return_unicode_hex_within_range("ba3", 2)
    na_na_pa = return_unicode_hex_within_range("ba8", 3)
    ma_yararavazhaLa_sa_ssa_sha_ha = return_unicode_hex_within_range("bae", 12)
    aa_e_ee_oo_ooo_connectors = return_unicode_hex_within_range("bbe", 5)
    a_aay_ai_connectors = return_unicode_hex_within_range("bc6", 3)
    o_oo_ou_stressing_connectors = return_unicode_hex_within_range("bca", 4)
    ou = return_unicode_hex_within_range("bd0", 1)
    ou_length_connector = return_unicode_hex_within_range("bd7", 1)
    numbers_and_misc_signs = return_unicode_hex_within_range("be6", 21)
    
    all_chars = ayudha_yezhuthu_stressing_connector + a_to_ooo + ye_to_i + o_O_ou_ka + nga_sa
    all_chars += ja + nya_ta + Na_tha + na_na_pa + ma_yararavazhaLa_sa_ssa_sha_ha
    all_chars += aa_e_ee_oo_ooo_connectors + a_aay_ai_connectors + o_oo_ou_stressing_connectors 
    all_chars += ou + ou_length_connector + numbers_and_misc_signs

    return all_chars

def return_kannada_unicode_isalnum():

    block1 = return_unicode_hex_within_range("c80", 13)
    block2 = return_unicode_hex_within_range("c8e", 3)
    block3 = return_unicode_hex_within_range("c8e", 23)
    block4 = return_unicode_hex_within_range("caa", 10)
    block5 = return_unicode_hex_within_range("cbc", 9)
    block6 = return_unicode_hex_within_range("cc6", 3)
    block7 = return_unicode_hex_within_range("cca", 4)
    block8 = return_unicode_hex_within_range("cd5", 2)
    block9 = return_unicode_hex_within_range("cdd", 2)
    block10 = return_unicode_hex_within_range("ce0", 4)
    block11 = return_unicode_hex_within_range("ce6", 10)
    block12 = return_unicode_hex_within_range("cf1", 3)

    all_chars = block1 + block2 + block3 + block4
    all_chars += block5 + block6 + block7 + block8
    all_chars += block9 + block10 + block11 + block12

    return all_chars

def check_unicode_block(character, unicode_block):
    # Returns True if `character` is inside `unicode_block`

    unicode_hex = "".join("{:02x}".format(ord(x)) for x in character)
    return unicode_hex in unicode_block

def get_en_unicode_sequence(token, unicode_hex):
    
    sequence = OrderedDict()
    num_eng, num_unicode = 0, 0
    get_count = lambda lang: str(num_eng) if lang=='en' else str(num_unicode)

    if check_unicode_block(token[0], unicode_hex):
        lang = 'uc'
        num_unicode += 1
    else:
        lang = 'en'
        num_eng += 1

    sequence[lang+"0"] = 0

    for idx, character in enumerate(list(token)[1:]):

        if check_unicode_block(character, unicode_hex):
            if lang == 'en':
                lang = 'uc'
                sequence[lang+get_count(lang)] = idx + 1
                num_unicode += 1
        else:
            if lang == 'uc':
                lang = 'en'
                sequence[lang+get_count(lang)] = idx + 1
                num_eng += 1

    sequence[lang+get_count(lang)] = len(token)
    return sequence

def tokenize_en_unicode_combinations(token_languages, token, english_reserved_token="ENG"):

    tokens_split, unicode_part = [], ""
    keys = list(token_languages.keys())
    for idx, key in enumerate(reversed(keys[:-1])):
        lang = "en" if "en" in key else "uc"
        start_of_lang_block = token_languages[key]
        end_of_lang_block = token_languages[keys[len(keys)-1 - idx]]
        
        if lang == "en":
            if end_of_lang_block - start_of_lang_block >= 3:
                if unicode_part == "":
                    tokens_split.append(english_reserved_token)
                else:
                    tokens_split.extend([unicode_part, englist_reserved_word])
                    unicode_part = ""
        else:
            unicode_part = token[start_of_lang_block:end_of_lang_block] + unicode_part
    
    if unicode_part != "":
        tokens_split.append(unicode_part)
    else:
        # no tamil characters means <=2 character english token
        tokens_split.append(english_reserved_token)

    tokens_split = list(reversed(tokens_split))

    return tokens_split

def return_en_unicode_tokenized_sentence(sentence, unicode_hex, english_reserved_token):

    tokens = sentence.split(' ')
    
    for token_index, token in enumerate(tokens):

        if not token in string.punctuation:
            
            token_languages = get_en_unicode_sequence(token, unicode_hex)
            token_replacement = tokenize_en_unicode_combinations(token_languages, token, \
                                        english_reserved_token=english_reserved_token)
            
            if token_replacement[0] == token:
                continue
            else:
                new_sentence_tokens = tokens[:token_index] + token_replacement + tokens[token_index+1:]
                sentence = " ".join(new_sentence_tokens)
    
    return sentence
