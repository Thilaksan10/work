from decimal import Decimal
import hashlib

class Blockbody:
    """
    Blockbody stores the amount of transactions in the block 
    as a counter and a list of verified transactions made 
    in the time between the creation of the previous block 
    and the current block.  
    """

    def __init__(self, transactions):
        """
        Constructor
        transactions: List<Transaction>
        """
        self.counter = 0
        while self.counter < len(transactions):
            self.counter += 1
        self.transactions = transactions

    def hash(self):
        """
        creates the hash of all transactions stored 
        in the block
        """
        h = hashlib.sha256()
        h.update(str(self.transactions).encode('utf-8')) 


    def __str__(self):
        transactions = 'Total Transactions: ' + str(self.counter) + '\n'
        transaction_fee = round(Decimal(0.0),2)
        transaction_amount = round(Decimal(0.0),2)
        for transaction in self.transactions:
            round_to = len(str(transaction.transaction_amount))
            transaction_fee = round(transaction_fee + transaction.fee,round_to)
            # print(transaction_fee)
            transaction_amount = round(transaction_amount + transaction.transaction_amount, round_to)
            transactions += str(transaction) + '\n\n'

        
        return transactions + 'Total Amount: ' + str(transaction_amount) + '             Total Fee: ' + str(transaction_fee) + '\n'      

