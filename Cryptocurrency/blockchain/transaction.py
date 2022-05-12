import datetime
import hashlib
from Crypto.Random import get_random_bytes
from decimal import Decimal

class Transaction:
    """
    Transaction stores a transaction amount, a hash 
    of the sender id, a hash of the receiver id, a 
    random 64 byte-long key and a timestamp, which stores
    the exact time the transaction was made
    """

    def __init__(self,sender_id,receiver_id,transaction_amount,is_reward):
        """
        Contrsuctor
        sender_id: int
        receiver_id: int
        transaction_amount: float
        """
        self.is_reward = is_reward
        self.timeStamp = datetime.datetime.now()
        self.key = get_random_bytes(64)
        # print(self.key)
        self.sender_id = self.hash_id(sender_id)
        self.receiver_id = self.hash_id(receiver_id)
        round_to = len(str(transaction_amount))
        if not self.is_reward:
            self.transaction_amount = round(Decimal(transaction_amount * 0.99999),round_to)
            self.fee = round(Decimal(transaction_amount) - self.transaction_amount,round_to)
        else:
            self.transaction_amount = round(Decimal(transaction_amount),round_to)   
            self.fee = round(Decimal(0.0),round_to)

        # print(self.transaction_amount)
        # print(self.fee)  
        self.key = self.hash_key() 
        # print(self.key)

    def calc_fee(self,transaction_amount):
        # self.transaction_amount = self.transaction_amount * 100000000000000
        pass
       

    def hash_id(self,id):
        """
        hashes the id with the key and the timstamp of the transaction.

        Arguments:
        id: int -- the id of the sender or receiver of the transaction

        Return:
        h.hexidigest: String
        """
        h = hashlib.sha256()
        h.update(
            str(id).encode('utf-8') + 
            str(self.key).encode('utf-8') + 
            str(self.timeStamp).encode('utf-8')
        )
        return h.hexdigest()

    def hash_key(self):
        """
        hashes the key of the transaction

        Return:
        h.hexidigest: String
        """
        h = hashlib.sha256()
        h.update(str(self.key).encode('utf-8'))
        return h.hexdigest()

    def __str__(self):
        return 'From: ' + str(self.sender_id) + '\nTo: ' + str(self.receiver_id) + '\nAmount: ' + str(self.transaction_amount) + ' Marecoin \nFee: ' + str(self.fee) + ' Marecoin \nDate: ' + str(self.timeStamp)  
