#subagent main
#startblock type: action
#timestamp 1740912853.8571844
#time_remaining 21552.925775289536 seconds

def combat_and_navigate(subagent):
    """Engage kobold with wand then navigate northern corridor to eastern branch."""
    nethack = subagent.tools['nethack']
    # Zap eastward at kobold (z + l)
    nethack.send_keys("zl")
    # Move north through corridor (k repeated 8x)
    nethack.send_keys("k"*8) 
    # Turn east at junction
    nethack.send_keys("l")
    # Move south down eastern branch (j repeated 4x)
    nethack.send_keys("jjjj")
    return True

self.add_action("Engage kobold & navigate eastern branch", combat_and_navigate)
#endblock
