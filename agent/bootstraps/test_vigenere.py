import unittest
from vigenere import vigenere_encrypt, vigenere_decrypt

class TestVigenereCipher(unittest.TestCase):

    def test_encryption(self):
        plaintext = "attackatdawn"
        key = "LEMONLEMONLE"
        expected_ciphertext = "LXFOPVEFRNHR"
        encrypted = vigenere_encrypt(plaintext, key)
        self.assertEqual(encrypted, expected_ciphertext)

    def test_decryption(self):
        ciphertext = "LXFOPVEFRNHR"
        key = "LEMONLEMONLE"
        expected_plaintext = "attackatdawn"
        decrypted = vigenere_decrypt(ciphertext, key)
        self.assertEqual(decrypted, expected_plaintext)

if __name__ == '__main__':
    unittest.main()
