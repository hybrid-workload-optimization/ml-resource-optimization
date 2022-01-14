import asyncio

async def async_main(coro):
    await coro

class AsyncWrapper():

    def run(self, coro):
        loop = asyncio.new_event_loop()
        loop.run_until_complete(async_main(coro["target"](coro['param1'], coro['param2'])))
        loop.close()