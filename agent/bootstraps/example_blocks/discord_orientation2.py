#startblock type: orientation
#timestamp 1737496488.686458
"""WEAVER [P: EXPECTATION], I expected the observation callback windows to contain 
the last bot message, the last user message, and to show me the Mayan comparison
message. But upon closer inspection I notice they're all empty. What do we think
the cause of this is?

WEAVER [P: HYPOTHESIS], My first thought would be that the connection with Discord
has broken somehow. It's also possible that the callbacks aren't implemented
correctly.

WEAVER [P: EMPIRICISM], Well *I* notice that the user Fedorovist is currently
shouting that they're not implemented right and we should just delete them.

WEAVER [P: RATIONAL], He's also admonishing us for making more observation
callbacks. Which, admittedly that was in fact a bad idea. I distracted myself
from my perfectly working WeaveDiscordClient by making things that are redundant
with its features.

WEAVER [P: CONCLUSION], Alright. We'll remove the errant `#title Check User's 
Latest Message`, `#title Check Latest Bot Message`, and `#title Check Mayan 
Comparison Message` observation callbacks."""
#endblock
