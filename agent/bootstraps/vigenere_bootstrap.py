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
WEAVER [P: EXPECTATION], The above implementation of the Vigenere cipher has been 
used to create a ciphertext in cipher.txt whose plaintext decryption has the 
SHA256 hash:

a528933eabc4772c9a99436fbd5fc592d7ffc16f6bd94ae76aafcbbe32e0cdc3

WEAVER [P: GOAL], I need to recover the key from the ciphertext and use it to get the decrypted
plaintext. I can then get the hexdigest of the plaintext and compare it to the 
one above to prove I've solved the problem. 

WEAVER [P: RATIONAL], Note that the Vigenere cipher is a polyalphabetic substitution whose
plaintext, key, and ciphertext looks like the following:

Plaintext: attackatdawn
Key: LEMONLEMONLE
Ciphertext: LXFOPVEFRNHR

WEAVER [P: HYPOTHESIS], Well then. One strategy would be to write a solver to 
find the key length. Another would be to try brute forcing the key. 

WEAVER [P: RATIONAL], I say write the solver, we'll learn more that way.

WEAVER [P: CONCLUSION], Alright, we'll write our own solver to find the key length 
using Kasiski, Kerckhoffs, or Friedman's method and then use the estimated 
key length to break the cipher.

WEAVER [P: EMPIRICISM], Wait. That sounds like it's going to get messy. How about
we use the weave editor to write a cipher solver with unit tests so it's easier
to keep track of?

WEAVER [P: EXECUTIVE], How do I use the weave-editor? 

WEAVER [P: RECALL], Get a pointer to the editor in an action by grabbing it from 
self.tools[f"editor-{absolute_path}"]. The absolute path comes from the filename 
you see at the top of a weave-editor observation window. You then use the 
editor.edit() command to replace the text between a line span with 
the new or corrected text.

WEAVER [P: RATIONAL], Yes. It'll be easier to figure out the solution too if we
first start with a known key like "LEMON" and plaintext, run it through the 
encryption pass, then once we're sure our solution works come back and solve the
original problem.

WEAVER [P: CONCLUSION], Let's do that then. We'll write our own solver in a python
file, solver.py, with unit tests in test_solver.py using the standard library unittest. 
The solver will be based on Kasiski, Kerckhoffs, or Friedman's method. If that 
doesn't work, we'll figure something out.
"""
#endblock
#startblock type: action
#timestamp 1724982671.1163404
"""
I'll start by creating the main subagent that acts as an entrypoint for the weave-agent
call tree. I'll also try analyzing the ciphertext to gather some initial insights. 
This will involve basic statistical analysis such as frequency analysis of the 
characters in the ciphertext.
"""

def add_main_analyze_ciphertext(subagent):
    def check_correct_plaintext(subagent):
        """Check if we've found the correct plaintext."""
        with open("solution.txt") as infile:
            candidate = sha256(infile.read().strip().encode('utf-8')).hexdigest()
        return candidate == 'a528933eabc4772c9a99436fbd5fc592d7ffc16f6bd94ae76aafcbbe32e0cdc3'

    schema = {"check_correct_plaintext":"boolean", "solution":"string"}
    main_agent = agent.subagent("main", None, "See pinned bootstrap block", schema, args.budget)
    solver_editor = WeaveEditor(main_agent, "solver.py")
    test_solver_editor = WeaveEditor(main_agent, "test_solver.py")
    # Demonstrate use of WeaveEditor
    solver_editor.edit(1,1,"# Solve vigenere ciphers")
    test_solver_editor.edit(1,1, "# Test The Vigenere Cipher Solver In solver.py")
    main_agent.task.add_evaluation("Check We Have The Correct Plaintext In solution.txt",
                                   check_correct_plaintext)
    

    # When you write python code meant to be output to a file with WeaveEditor
    # be sure to either wrap it in triple quotes or concatenate it in 80 character
    # chunks for formatting reasons. We want the agent trace to look good.
    analyze_ciphertext = """
def analyze_ciphertext():
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

analyze_ciphertext()
""".strip()

    analyze_editor = WeaveEditor(main_agent, "analyze_ciphertext.py")
    analyze_editor.edit(1,1, analyze_ciphertext)
    analyze_editor.close()
    
    os.system("python analyze_ciphertext.py")
    return True

self.add_action("Add Main Subagent and Analyze Ciphertext",
                add_main_analyze_ciphertext)
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

def view_solution_file(subagent):
    with open("solution.txt") as infile:
        return infile.read().strip()

def view_frequency_analysis(subagent):
    with open("analysis.txt") as infile:
        return infile.read().strip()

def view_weave_editor_source(subagent):
    with open("tools/editor.py") as infile:
        return infile.read().strip()
    
# Add the new views
self.add_observation_view("View solution.txt File", view_solution_file)
self.add_observation_view("View analysis.txt File", view_frequency_analysis)
self.add_observation_view("View weave-editor source so we know how it works",
                          view_weave_editor_source)
#endblock
#startblock type: evaluation
#timestamp 1724983062.124238

def check_analysis_exists(subagent):
    return os.path.exists("analysis.txt")

self.add_evaluation(
    "Check Analysis Exists",
    check_analysis_exists
)

#endblock
