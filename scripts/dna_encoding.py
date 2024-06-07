def text_to_dna(text):
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    binary_str = ''.join(format(ord(char), '08b') for char in text)
    dna_str = binary_str.replace('0', 'A').replace('1', 'T')
    return dna_str

def dna_to_text(dna_str):
    if not isinstance(dna_str, str):
        raise ValueError("Input must be a string")
    binary_str = dna_str.replace('A', '0').replace('T', '1')
    text = ''.join(chr(int(binary_str[i:i+8], 2)) for i in range(0, len(binary_str), 8))
    return text

def number_to_dna(number):
    if not isinstance(number, (int, float)):
        raise ValueError("Input must be a number")
    binary_str = format(int(number * 10), '032b')  # Using 32 bits for numbers
    dna_str = binary_str.replace('0', 'A').replace('1', 'T')
    return dna_str

def dna_to_number(dna_str):
    if not isinstance(dna_str, str):
        raise ValueError("Input must be a string")
    binary_str = dna_str.replace('A', '0').replace('T', '1')
    number = int(binary_str, 2)
    return number / 10  # Scale back the rating

def encode_data(row):
    try:
        encoded = {}
        encoded['Book Name'] = text_to_dna(row['Book Name'])
        encoded['Author Name'] = text_to_dna(row['Author Name'])
        encoded['Rating'] = number_to_dna(int(row['Rating'] * 10))  # Assume rating is scaled by 10
        encoded['Price'] = text_to_dna(row['Price'])  # Prices are in string format
        return encoded
    except Exception as e:
        print(f"Error encoding data: {e}")
        return None

def decode_data(encoded_row):
    try:
        decoded = {}
        decoded['Book Name'] = dna_to_text(encoded_row['Book Name'])
        decoded['Author Name'] = dna_to_text(encoded_row['Author Name'])
        decoded['Rating'] = dna_to_number(encoded_row['Rating'])
        decoded['Price'] = dna_to_text(encoded_row['Price'])
        return decoded
    except Exception as e:
        print(f"Error decoding data: {e}")
        return None
