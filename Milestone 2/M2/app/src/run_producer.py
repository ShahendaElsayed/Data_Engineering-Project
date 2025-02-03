import docker
from kafka import KafkaProducer

import time

def wait_for_kafka(broker, retries=5, delay=10):
    """
    Waits for Kafka to become available by attempting to create a producer connection.
    """
    for i in range(retries):
        try:
            producer = KafkaProducer(bootstrap_servers=broker)
            print("Connected to Kafka!")
            producer.close()  # Close the connection after verifying
            return True
        except Exception as e:
            print(f"Retry {i+1}/{retries}: Kafka not available, waiting...")
            time.sleep(delay)
    raise Exception("Kafka broker not available after retries.")


def start_producer(id,  topic_name='fintech'):
  kafka_broker = "kafka:9092"
  #wait_for_kafka(kafka_broker)
  docker_client = docker.from_env()
  container = docker_client.containers.run(
    "mmedhat1910/dew24_streaming_producer",
    detach=True,
    name=f"m2_producer_container_{int(time.time())}",
    environment={
      "ID": id,
      "KAFKA_URL":"kafka:9092",
      "TOPIC":topic_name,
      'debug': 'True'
    },
    network='m2_default',
    volumes={
        '/var/run/docker.sock':{
        'bind':'/var/run/docker.sock',
        'mode':'rw'
      }
    }
  )


  print('Container initialized:', container.id)
  return container.id

def stop_container(container_id):
  docker_client = docker.from_env()
  container = docker_client.containers.get(container_id)
  container.stop()
  container.remove()
  print('Container stopped:', container_id)
  return True