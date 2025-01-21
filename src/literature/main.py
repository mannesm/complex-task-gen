from client.zotero_client import ZoteroClient
from client.openai_client import gpt_chat_completion
zotero_client = ZoteroClient().client


items = ZoteroClient().get_top_items(limit=1)
# zotero_client.items()

# we've retrieved the latest five top-level items in our library
# we can print each item's item type and ID
for item in items:
    print("Item: %s | Key: %s | ID: %s" % (item["data"]["itemType"], item["data"]["key"], item["library"]["id"]))
    print(item["data"]["abstractNote"])
    print(item["data"]["title"])


gpt_chat_completion(item["data"]["abstractNote"])

id = str(item["library"]["id"])

zotero_client.dump(itemkey="2GGX7REV", filename='test')


# print(zotero_client.get_item(16183954))
#
