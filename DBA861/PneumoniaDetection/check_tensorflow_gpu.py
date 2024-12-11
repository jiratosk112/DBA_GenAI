import tensorflow as tf 

import time
from datetime import timedelta

start_time = time.time()

print (f"\nPhysical Devices: {tf.config.list_physical_devices()}")

print (f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

end_time = time.time()

elapsed_time  = end_time - start_time
print(f"\nElapsed Time: {str(timedelta(seconds=elapsed_time))}")
