import os
import dotenv
from pyzotero import zotero
from constants import ZOTERO_USER_ID, ZOTERO_LIBRARY_TYPE

class ZoteroClient:
    def __init__(self):
        dotenv.load_dotenv()
        self.api_key = os.getenv("ZOTERO_KEY")
        self.client = zotero.Zotero(
            library_id=ZOTERO_USER_ID,
            library_type=ZOTERO_LIBRARY_TYPE,
            api_key=self.api_key
        )

    def get_top_items(self, limit: int=5):
        """
        Get the top N items from the Zotero library
        :param limit: Number of items to retrieve
        :return:
        """
        return self.client.top(limit=limit)

    def get_item(self, item_key: str):
        """
        Get the item with the given key
        :param item_key: Key of the item to retrieve
        :return:
        """
        return self.client.item(item_key)
