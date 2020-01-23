# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 12:45:01 2020

@author: Raj
"""


'''
Script to constantly poll a folder of data and generate an image from that
'''

import time
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from ffta import pixel, pixel_utils

class MyHandler(FileSystemEventHandler):
    
    def __init__(self):
        self.count = 0
    
    def on_created(self, event):
        
        if event.is_directory:
            print('asdsajld;')
        else:
            print(event.src_path)
            self.count += 1
   
     
if __name__ == '__main__':
    path_to_watch = sys.argv[1]
    
    print('Loading data from ', path_to_watch)
    my_observer = Observer()
    my_event_handler = MyHandler()
    my_observer.schedule(my_event_handler, path_to_watch, recursive=False)
    my_observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        my_observer.stop()
        my_observer.join()
        print('Stop')
        print(my_event_handler.count)
    