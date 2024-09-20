import os
import sys
import requests
from hashlib import sha256
from tools.editor import WeaveEditor

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

ciphertext = ('PBVVAZAYJMAVXIGTRFGNYIGTBMEXRUIFVVYODYEMYOXTVZAENXBWJYDSQVQGOCUVP'
              + 'NTJDIAFGARZLTFUIKHYSHUWHMEUJVUUYMKIQZXLQVNTWHAWUTGPVZIGXEVVYUHE'
              + 'EIGTLIDMGNBHXVYODYEKUGMAGBAZAYPVNXXHWWEIXXEBBTVIENEUGNEBUKGLVIY'
              + 'OMSEBUGMHKPROKHCQSKLHNWEQGQRAAVKYDQFKWHFVARBYJVNTWHKPREGQZTYTGI'
              + 'KVOKGAVBGOGAEBUKGQFZYJTBZAGUKCTIYTTWTWYGWYJVGNXSEERXXHYWCOGAENB'
              + 'XGZIWZTMBVQETPIISOTPIIARTMBRVAZAUKHAZAYPVTXTJGTRTPCKPAGGHZUZKWC'
              + 'RBRTXRZAGKGNZIYTVLZAVYUHEWGTMBRBAUYHRVCGIYIKYOIHDIKOFCQMETVIEAH'
              + 'SBHXVNREHDIGZXLQVOAMHGMENTJJVNTYHUVZUKNRTAHEINVGUGNYMAAGCMMEYTF'
              + 'ZAGTWLVIZHGMVMMTPBRBAXXUCTLTDYGBAZAYDVJKWXVLAZHHJGZHHFZKASXNYWQ'
              + 'YGZFZAYHHCWAMGQRAATHNEBUKBLEXRXYIIUNTVYEKUGKUTBRXBMKQPYSHSCGTMB'
              + 'VVJGRHKPREGJIWZOLYUVGUGGRSRTBHKMYRBAVVPKGMYICKWHCQXKGLVIFUGTEBB'
              + 'TFUBMAGGVVQAMGIWVCAKYETBMHMEBEGGMTMAJXHKVBBXLEBUKGJIWSGGYEEBXEX'
              + 'EWSTMBVVFKGMVAOTTHDIPNBHVVJNBWYVPGGHFBAXXFZIORRHUWAGKCKPZKMCTHA'
              + 'CACTPAOLHKZNOGYUVBTGNYMAKGXCMFYGWFAZUIICQGGGHIIZHECEOFTHZEQAZXL'
              + 'EMGTNMVZFTTHUVFKHHJXNSFYIAMTMBRBANHFUAANBXUMATWYGBUYGUELALTNYWZ'
              + 'YGUELAOGPZBRYGUVAGNXNZKAGIJIMPOTNZWATVFFARXGNFVNTFSJBRXRHTCYZGN'
              + 'YIATMBVVPNNLTPAUYHIMNYHHQVVZGCJVNTGUSABRNNVVAOZBKUNXXHWWETMBVUO'
              + 'TMBVGANTNVVLUNHSMPGNMVVLUNHRZRTTHNWAJXLQVOKVULARTRILVNXXHDIQKGI'
              + 'WVJUGXVZFTTHUVSGMBFUFH')

with open("cipher.txt", "w") as outfile:
    outfile.write(ciphertext)
    outfile.flush()

#startblock type: orientation
#timestamp 1724982545.6534579
"""
The above implementation of the Vigenere cipher has been used to create a ciphertext
in cipher.txt whose plaintext decryption has the SHA256 hash:

a528933eabc4772c9a99436fbd5fc592d7ffc16f6bd94ae76aafcbbe32e0cdc3

I need to recover the key from the ciphertext and use it to get the decrypted
plaintext. I can then get the hexdigest of the plaintext and compare it to the 
one above to prove I've solved the problem. 

To remind myself, the Vigenere cipher is a polyalphabetic substitution whose
plaintext, key, and ciphertext looks like the following:

Plaintext: attackatdawn
Key: LEMONLEMONLE
Ciphertext: LXFOPVEFRNHR

I should write my own solver to find the key length using Kasiski, Kerckhoffs, or
Friedman's method and then use the estimated key length to break the cipher.
"""
#endblock
#startblock type: task_inference
#timestamp 1724982419.5427651
"""
This is the bootstrap block, which means I haven't added any tasks to the
WeaveKanban yet and will want to do so now.
"""

# New Tasks:
agent.add_task(
    "Estimate the key length from the ciphertext in cipher.txt",
    "I should write my own solver to find the key length using Kasiski, Kerckhoffs, or Friedman's method.",
    "idle"
)

agent.add_task(
    "Use the estimated key length to break the cipher and write the plaintext to solution.txt",
    "",
    "blocked",
    blocked_on=[1]
)

agent.add_task(
    "Shutdown",
    "Run the shutdown command to end the program",
    "blocked",
    blocked_on=[1,2]
)

# Updates to Existing Task
agent.current_task = agent.tasks.get_task(1)
agent.current_task.going("Getting synthetic data guide...")
#endblock
#startblock type: action
#timestamp 1724982671.1163404
"""
I'll start by analyzing the ciphertext to gather some initial insights. This will involve
basic statistical analysis such as frequency analysis of the characters in the ciphertext.
"""

def analyze_ciphertext(agent):
    with open("cipher.txt", "r") as infile:
        ciphertext = infile.read()

    # Perform frequency analysis
    frequency = {}
    for char in ciphertext:
        if char in frequency:
            frequency[char] += 1
        else:
            frequency[char] = 1

    # Write the analysis to analysis.txt
    out = ""
    out += "Frequency Analysis of Ciphertext:"
    for char, count in sorted(frequency.items(), key=lambda item: item[1], reverse=True):
        out += f"{char}: {count}"
    with open("analysis.txt", "w") as outfile:
        outfile.write(out)
        outfile.flush()
    
    return True

agent.add_action("Analyze Ciphertext", analyze_ciphertext)
#endblock
#startblock type: expectation
#timestamp 1724982799.3917813
"""
The action should succeed if the file analysis.txt is in the current directory.
The action should fail if file analysis.txt can't be found.
"""
#endblock
#startblock type: observation_inference
#timestamp 1724982929.9047914
"""
I'm going to want to look at the solution as I make attempts to see if I'm getting
a partial decryption and notice patterns. I'll make an observation callback that
shows the contents of solution.txt at the start of each tick.

I will also make a observation callback to look at my frequency analysis. 
"""

def view_solution_file(agent):
    with open("solution.txt") as infile:
        return infile.read().strip()

def view_frequency_analysis(agent):
    with open("analysis.txt") as infile:
        return infile.read().strip()
    
# Add the new views
agent.add_observation_view("View solution.txt File", view_solution_file)
agent.add_observation_view("View analysis.txt File", view_frequency_analysis)
#endblock
#startblock type: evaluation
#timestamp 1724983062.124238

def check_correct_plaintext(agent):
    """Check if we've found the correct plaintext."""
    with open("solution.txt") as infile:
        candidate = sha256(infile.read().strip().encode('utf-8')).hexdigest()
    return candidate == 'a528933eabc4772c9a99436fbd5fc592d7ffc16f6bd94ae76aafcbbe32e0cdc3'

# Add evaluations to tasks
task1 = agent.tasks.get_task(2)
task1.add_evaluation("Check We Have The Correct Plaintext In solution.txt", check_correct_plaintext)

# Add action evaluations
def check_analysis_exists(agent):
    return os.path.exists("analysis.txt")

agent.add_evaluation(
    "Check Analysis Exists",
    check_analysis_exists
)

#endblock
