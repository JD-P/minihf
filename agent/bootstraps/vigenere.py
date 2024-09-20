def vigenere_encrypt(plaintext, key):
    encrypted_text = []
    key_length = len(key)
    key_as_int = [ord(i) - 65 for i in key.upper()]
    plaintext_int = [ord(i) - 97 for i in plaintext.lower()]
    for i in range(len(plaintext_int)):
        value = (plaintext_int[i] + key_as_int[i % key_length]) % 26
        encrypted_text.append(chr(value + 65))
    return "".join(encrypted_text)

def vigenere_decrypt(ciphertext, key):
    decrypted_text = []
    key_length = len(key)
    key_as_int = [ord(i) - 65 for i in key.upper()]
    ciphertext_int = [ord(i) - 65 for i in ciphertext.upper()]
    for i in range(len(ciphertext_int)):
        value = (ciphertext_int[i] - key_as_int[i % key_length]) % 26
        decrypted_text.append(chr(value + 97))
    return "".join(decrypted_text)

if __name__ == "__main__":
    # Example usage:
    plaintext = "attackatdawn"
    key = "LEMONLEMONLE"

    encrypted = vigenere_encrypt(plaintext, key)
    print(f"Encrypted: {encrypted}")

    decrypted = vigenere_decrypt(encrypted, key)
    print(f"Decrypted: {decrypted}")
