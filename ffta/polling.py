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
from ffta import pixel, line, pixel_utils
from matplotlib import pyplot as plt
import argparse

class MyHandler(FileSystemEventHandler):
    
    def __init__(self):
        self.count = 0
        
    def on_created(self, event):
        
        if not event.is_directory:
            path = event.src_path.split('\\')
            print('Loading '+ path[-1])
            
            params_file = r'/'.join(path[:-1]) + r'\parameters.cfg'
            
            npixels, parameters = pixel_utils.load.configuration(params_file)
            signal = pixel_utils.load.signal(event.src_path)
            print(parameters)
            this_line = line.Line(signal, parameters, npixels)
            
            self.count += 1
   
     
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path where data are being saved')
    
    path_to_watch = parser.parse_args().path
    print('Loading data from ', path_to_watch)
    
    params_file = path_to_watch + r'\parameters.cfg'

    my_observer = Observer()
    my_event_handler = MyHandler()
    my_observer.schedule(my_event_handler, path_to_watch, recursive=False)
    my_observer.start()
    fig, ax = plt.subplots()
    ax.plot([0,1,2], [3,4,3], 'k--')
    a = 1
    plt.show()
    try:
        while True:
            time.sleep(1)
            a += 1
            ax.plot(a*[0,1,2], a*[3,4,3], 'k--')
            plt.show()
    except KeyboardInterrupt:
        my_observer.stop()
        my_observer.join()
        print('Stop')
    print(my_event_handler.count)
    