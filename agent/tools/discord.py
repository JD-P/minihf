# This one is probably non-obvious how you're supposed to set it up so lemme give
# some pointers.
#
# While logged into Discord go to this URL: https://discord.com/developers/applications
#
# Make a new application with the name of your bot, etc.
#
# Go to "Installation", there should be a URL labeled "install link" that looks like
# https://discord.com/oauth2/authorize?client_id=NUMBERS which has your client ID.
#
# Append &permissions=137439317056&scope=bot to the end of that URL and visit it,
# you should be prompted to add the bot you've made to your server.
#
# Then add to the agent folder a file called `discord.json`. It should have the
# key 'key' with the security token you get on the "Bot" page of the applications
# interface (NOT "OAuth2").  Under the key 'cid' should be the channel ID of the
#channel in the server you want the bot to join. So:
#
# {"key":AUTH_TOKEN_NUMBERS_AND_LETTERS, "cid":CHANNEL_ID_NUMBERS}
#
# You can get the channel ID by turning on developer mode under
# "Advanced" in Discord settings. You then go to the channel you want the cid
# for, right click it, and choose "Copy Channel ID".

import json
from datetime import datetime
import multiprocessing
import nextcord
intents = nextcord.Intents.default()
intents.message_content = True
from nextcord.ext import commands
import requests
import asyncio
from aiohttp import web

class DiscordBotServer:
    def __init__(self, token, channel_id):
        self.token = token
        self.channel_id = int(channel_id)
        self.bot = commands.Bot(command_prefix="!", intents=intents)
        self.messages = []

        @self.bot.event
        async def on_ready():
            print(f'Logged in as {self.bot.user}')
            self.channel = self.bot.get_channel(self.channel_id)

        @self.bot.event
        async def on_message(message):
            if message.channel.id == self.channel_id:
                self.messages.append(message)

    async def start_bot(self):
        await self.bot.start(self.token)

    async def send_message(self, content):
        await self.channel.send(content)

    async def reply_to_message(self, message_id, content):
        message = await self.channel.fetch_message(message_id)
        await message.reply(content)

    async def react_to_message(self, message_id, emoji):
        message = await self.channel.fetch_message(message_id)
        await message.add_reaction(emoji)

    async def get_messages(self):
        return [{"author": msg.author.name,
                 "content": msg.content,
                 "timestamp":msg.created_at.timestamp()}
                for msg in self.messages]

    async def handle_request(self, request):
        data = await request.json()
        action = data.get("action")
        if action == "send_message":
            await self.send_message(data["content"])
        elif action == "reply_to_message":
            await self.reply_to_message(data["message_id"], data["content"])
        elif action == "react_to_message":
            await self.react_to_message(data["message_id"], data["emoji"])
        elif action == "get_messages":
            messages = await self.get_messages()
            return web.json_response(messages)
        return web.Response(text="OK")

    async def start_server(self):
        app = web.Application()
        app.router.add_post('/', self.handle_request)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', 8080)
        await site.start()

    async def run(self):
        await asyncio.gather(self.start_bot(), self.start_server())

def run_server(token, channel_id):
    bot_server = DiscordBotServer(token, channel_id)
    asyncio.run(bot_server.run())

class WeaveDiscordClient:
    def __init__(self, agent, token, channel_id, server_url="http://localhost:8080"):
        self.agent = agent
        self.channel_id = channel_id
        self.server_url = server_url
        self.agent.tools[f"discord-bot-{self.channel_id}"] = self
        self.observation_view = {"type": "observation",
                                 "title": f"WeaveDiscordClient (discord-bot-{self.channel_id})",
                                 "tool":f"discord-bot-{self.channel_id}",
                                 "callback": self.render}
        self.agent.add_observation_view(
            f"WeaveDiscordClient (discord-bot-{self.channel_id})",
            self.render,
            tool=f"discord-bot-{self.channel_id}"
        )

        # Start the server in a separate process
        self.server_process = multiprocessing.Process(target=run_server, args=(token, channel_id))
        self.server_process.start()

    def render(self, agent):
        response = requests.post(self.server_url, json={"action": "get_messages"})
        messages = response.json()
        rendered_text = "'''Messages:\n"
        for msg in messages:
            dt = datetime.fromtimestamp(msg["timestamp"])
            formatted_time = dt.strftime('%Y-%m-%d %H:%M')
            rendered_text += f"{formatted_time} <{msg['author']}>: {msg['content']}\n"
        rendered_text += "'''"
        return rendered_text

    def get_messages(self):
        response = requests.post(self.server_url, json={"action": "get_messages"})
        return response.json()
    
    def send_message(self, content):
        requests.post(self.server_url, json={"action": "send_message", "content": content})

    def reply_to_message(self, message_id, content):
        requests.post(self.server_url, json={"action": "reply_to_message", "message_id": message_id, "content": content})

    def react_to_message(self, message_id, emoji):
        requests.post(self.server_url, json={"action": "react_to_message", "message_id": message_id, "emoji": emoji})

    def close(self):
        self.server_process.terminate()
        self.server_process.join()
        del self.agent.tools[f"discord-bot-{self.channel_id}"]
        self.agent.remove_observation_view(self.observation_view)

# Example usage
if __name__ == "__main__":
    token = 'your_token'
    channel_id = 'your_channel_id'
    agent = None  # Replace with your actual agent instance
    client = WeaveDiscordClient(agent, token, channel_id)

    # Keep the main process running
    try:
        while True:
            pass
    except KeyboardInterrupt:
        client.close()
