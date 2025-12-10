import re

class IPATokenizer:
    """
    Universal IPA Tokenizer for PsycheTTS.
    Maps IPA characters to integer IDs.
    """
    def __init__(self):
        # Basic IPA symbols (Common across EN/SI/etc)
        self.ipa_symbols = [
            '<pad>', '<unk>', '<bos>', '<eos>', ' ',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            'ɑ', 'ɒ', 'æ', 'ə', 'ɛ', 'ɜ', 'ɪ', 'i', 'ŋ', 'ɔ', 'u', 'ʊ', 'ʌ', 'ʒ', 'θ', 'ð', 'ʃ', 'tʃ', 'dʒ', 'ʔ', 'ː', 'ʰ', 'ʲ', 'ˈ', 'ˌ', '.',
            # Sinhala specific nuances if any (often covered above)
            'ɖ', 'ʈ', 'Pb', 'nd', 'ng' # Prenasalized
        ]
        self.char_to_id = {s: i for i, s in enumerate(self.ipa_symbols)}
        self.id_to_char = {i: s for i, s in enumerate(self.ipa_symbols)}
        self.vocab_size = len(self.ipa_symbols)

    def encode(self, ipa_string):
        """String -> List[Int]"""
        ids = []
        for char in ipa_string:
            if char in self.char_to_id:
                ids.append(self.char_to_id[char])
            else:
                ids.append(self.char_to_id['<unk>'])
        return ids

    def decode(self, ids):
        """List[Int] -> String"""
        return "".join([self.id_to_char.get(i, "") for i in ids])


class SinhalaPhonemizer:
    """
    Rule-based G2P for Sinhala -> IPA.
    Sinhala is highly phonetic, allowing for direct mapping.
    """
    def __init__(self):
        self.mapping = {
            # Vowels
            'අ': 'ə', 'ආ': 'aː', 'ඇ': 'æ', 'ඈ': 'æː', 'ඉ': 'i', 'ඊ': 'iː', 'උ': 'u', 'ඌ': 'uː',
            'එ': 'e', 'ඒ': 'eː', 'ඔ': 'o', 'ඕ': 'oː',
            # Consonants
            'ක': 'k', 'ඛ': 'kʰ', 'ග': 'g', 'ඝ': 'gʰ', 'ඞ': 'ŋ',
            'ච': 'tʃ', 'ඡ': 'tʃʰ', 'ජ': 'dʒ', 'ඣ': 'dʒʰ', 'ඤ': 'ɲ',
            'ට': 'ʈ', 'ඨ': 'ʈʰ', 'ඩ': 'ɖ', 'ඪ': 'ɖʰ', 'ණ': 'ɳ',
            'ත': 't', 'ථ': 'tʰ', 'ද': 'd', 'ධ': 'dʰ', 'න': 'n',
            'ප': 'p', 'ඵ': 'pʰ', 'බ': 'b', 'භ': 'bʰ', 'ම': 'm',
            'ය': 'j', 'ර': 'r', 'ල': 'l', 'ව': 'ʋ', 'ශ': 'ʃ', 'ෂ': 'ʂ', 'ස': 's', 'හ': 'h', 'ළ': 'ɭ', 'ෆ': 'f',
            # Modifiers
            '්': '', # Virama (removes inherent vowel)
            'ා': 'aː', 'ැ': 'æ', 'ෑ': 'æː', 'ි': 'i', 'ී': 'iː', 'ු': 'u', 'ූ': 'uː',
            'ෙ': 'e', 'ේ': 'eː', 'ො': 'o', 'ෝ': 'oː', 'ෛ': 'ai', 'ෞ': 'au',
            'ං': 'ŋ', 'ඃ': 'h'
        }
        
    def phonemize(self, text):
        """
        Converts Sinhala text to IPA string.
        Logic: Iterate chars. Consonants have inherent 'ə'. Virama removes it. Vowel signs replace it.
        """
        ipa_output = []
        skip_next = False
        
        chars = list(text)
        for i, char in enumerate(chars):
            if skip_next:
                skip_next = False
                continue
                
            if char in self.mapping:
                # Handle base consonant
                ipa_val = self.mapping[char]
                
                # Check for vowel modifiers (next char)
                next_char = chars[i+1] if i+1 < len(chars) else None
                
                if self.is_consonant(char):
                    if next_char in self.mapping and self.is_vowel_sign(next_char):
                        # Consonant + Vowel Sign
                        ipa_output.append(ipa_val) # Consonant
                        ipa_output.append(self.mapping[next_char]) # Vowel
                        skip_next = True # Consumed vowel sign
                    elif next_char == '්': # Virama
                        ipa_output.append(ipa_val) # Just consonant, no inherent vowel
                        skip_next = True
                    else:
                        # Consonant with inherent vowel 'ə'
                        ipa_output.append(ipa_val)
                        ipa_output.append('ə')
                else:
                    # Independent Vowel or other symbol
                     ipa_output.append(ipa_val)
            else:
                # Punctuation or space
                ipa_output.append(char)
                
        return "".join(ipa_output)

    def is_consonant(self, char):
        return 'ක' <= char <= 'ෆ' # Rough range check
    
    def is_vowel_sign(self, char):
        return char in ['ා','ැ','ෑ','ි','ී','ු','ූ','ෙ','ේ','ො','ෝ']

class TextProcessor:
    def __init__(self):
        self.si_g2p = SinhalaPhonemizer()
        self.tokenizer = IPATokenizer()
        
    def process(self, text, lang='en'):
        """
        Returns (indices, raw_ipa)
        """
        if lang == 'si':
            ipa = self.si_g2p.phonemize(text)
        else:
            # Mock English G2P for now
            # In prod, subprocess.run(['espeak-ng', ...])
            ipa = text # Placeholder
            
        indices = self.tokenizer.encode(ipa)
        return indices, ipa

if __name__ == "__main__":
    # Test Sinhala G2P
    tp = TextProcessor()
    text = "අම්මා" # Amma (Mother)
    indices, ipa = tp.process(text, lang='si')
    print(f"Sinhala: {text} -> IPA: {ipa}")
    
    text2 = "මම ඔයාට ආදරෙයි" # Mama oyata adarei
    indices2, ipa2 = tp.process(text2, lang='si')
    print(f"Sinhala: {text2} -> IPA: {ipa2}")
