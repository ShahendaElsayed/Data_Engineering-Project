import pandas as pd
from kafka import KafkaConsumer
import json
import time
from cleaning import clean_consumer
from db import save_to_db

# while True:
#     message = consumer.poll(timeout_ms=2000)  
    
#     if message:
#         empty_poll_count = 0  # Reset count when a message arrives
#         for tp, messages in message.items():
#             for msg in messages:
#                 print(f"Received: {msg.value}")
#                 new_row = pd.DataFrame([msg.value], columns=['timestamp','id', 'name', 'age', 'city'])
#                 df = pd.concat([df, new_row], ignore_index=True)
#     else:
#         empty_poll_count += 1
#         print("No messages received, polling again...")

#         if empty_poll_count >= empty_poll_limit:
#             print("No messages after multiple polls. Exiting.")
#             break
# consumer.close()

def run_consumer():
    """Kafka consumer that processes messages."""
    consumer = KafkaConsumer(
        'fintech',
        bootstrap_servers='kafka:9092',
        auto_offset_reset='latest',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

    print("Listening for messages in 'fintech' topic...")
    
    while True:
        # Poll for messages
        message = consumer.poll(timeout_ms=2000)

        if message:
            for tp, messages in message.items():
                for msg in messages:
                    print(f"Received: {msg.value}")
                    row = msg.value
                    # top if EOF is received
                    if row == "EOF":
                        print("EOF received. Stopping consumer.")
                        consumer.close()
                        return
                    
                    # Clean the row
                    new_row = pd.DataFrame([msg.value])
                    cleaned_row = clean_consumer(new_row)
                    # Save the cleaned row to the database
                    save_to_db(cleaned_row)
        else:
            print("No messages received. Polling again...")

