import asyncio
import aiohttp
from collections import deque

token = open('../resources/token.key', 'r').readline()
query = deque()
query_ready = deque()


async def get_friends_ids(session, id):
    async with session.get(
            f'https://api.vk.com/method/friends.get?user_id={id}&order=random&fields=last_seen&access_token={token}&v=5.126') as response:
        json = await response.json()
        if resp_dict := json.get('response'):
            if resp_dict.get('count'):
                if items := resp_dict.get('items'):
                    for i in items:
                        if i.get('id'):
                            query.append(i.get('id'))
                            query_ready.append(i.get('id'))


async def main():
    first = '180145112'
    query.append(first)
    query_ready.append(first)
    async with aiohttp.ClientSession() as session:
        i = 0
        while i < 12:
            tasks = []
            while len(query):
                tasks.append(get_friends_ids(session, query.popleft()))
            await asyncio.gather(*tasks)
            i += 1


asyncio.run(main())

# query_ready - очередь пользователей, готовых для обработки

print(f'Найдено id: {len(query_ready)}')
print(query_ready)

# какие-то действия с query_ready
