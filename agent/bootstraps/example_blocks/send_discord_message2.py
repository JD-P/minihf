#startblock type: action
#timestamp 1737486187.765493
def send_feudal_system_info(subagent):
    # Take the CID from the observation callback window
    client = self.tools['discord-bot-1325039818673094739']
    message = (
        "The feudal system began to decline in the late Middle Ages and early "
        "modern period due to several factors. Economic changes, such as the "
        "growth of trade and towns, weakened the feudal economy. Political "
        "centralization and the rise of strong monarchies also contributed to "
        "its end. The system largely ended by the 16th century in most of Europe, "
        "although remnants persisted in some regions until the 19th century."
    )
    client.send_message(message)
self.add_action("Send feudal system info", send_feudal_system_info)
#endblock
