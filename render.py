
from MinePi_SkinGenerator import render_3d_skin, render_3d_head, render_3d_skin_random
import asyncio

async def main():
    #Render a full body skin
    im = await render_3d_skin_random("Player")
    im.show()

    #Render a head only skin
    # im = await render_3d_head("Herobrine")
    # im.show()

asyncio.run(main())