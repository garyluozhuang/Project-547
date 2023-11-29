from openai import OpenAI
import pandas as pd


def prompterVisual(title, description, url):
    prompt = "Create a simple, 2-3 sentence description of the following video game based on its amazon product data and this product image. The description should not be an advertisement for this product. The title of this product is: '%s', the description of this product is: '%s'" % (
        title, description)
    print("Prompt: " + prompt + "\n")
    print("Image URL: " + url + "\n")
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": url,
                        },
                    },
                ],
            }
        ],
        model="gpt-4-vision-preview",
        max_tokens=100
    )
    return response


def prompterNoVisual(title, description):
    prompt = "Create a simple, 2-3 sentence description of the following video game based on its amazon product data. The description should not be an advertisement for this product. The title of this product is: '%s', the description of this product is: '%s'" % (
        title, description)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ]
            },

        ],
        max_tokens=100
    )
    return response


def prompterJustTitle_Visual(title, url):
    prompt = "Create a simple, 2-3 sentence description of the following video game based on this product image. The description should not be an advertisement for this product. The title of this product is: '%s'" % (
        title)
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": url,
                        },
                    },
                ],
            }
        ],
        model="gpt-4-vision-preview",
        max_tokens=100
    )
    return response


if __name__ == '__main__':
    client = OpenAI(
        organization="ORG-ID,
        api_key="API-KEY"
    )
    # CHANGE LOCATION AS NEEDED
    video_games = pd.read_json('/Users/adambeauchaine/PycharmProjects/GPT4_Desc_Gen/Data/meta_Video_Games.json',
                               lines=True)
    # print(video_games.columns)
    # print(video_games.loc[1])
    i = 0
    j = 0
    while True:
        try:
            name = video_games.loc[i]['title']
            comm = str.strip(video_games.loc[i]['description'][0])
            url = video_games.loc[i]['imageURL'][0]
        except(IndexError):
            i += 1
            continue
        i += 1
        j += 1
        vRespArr = prompterVisual(name, comm, url)
        nvRespArr = prompterNoVisual(name, comm)
        jvRespArr = prompterJustTitle_Visual(name, url)
        print("GPT4 Response: " + nvRespArr.choices[0].message.content + "\n")
        print("GPT4-V Response (Image and text): " + vRespArr.choices[0].message.content + "\n")
        print("GPT4-V Response (Just title + image): " + jvRespArr.choices[0].message.content + "\n")
        if j == 10:
            exit(0)
