
import argparse
import asyncio
import time
import numpy as np
import torch
import cv2

from model import overparameterized_model

parser = argparse.ArgumentParser(description='')
parser.add_argument("--vid_path", type=str, help = ' the path of video')
args = parser.parse_args()

class DEMO():

    def __init__(self, vid_path):
        torch.cuda.current_device()
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = overparameterized_model()
        checkpoint = torch.load("model.pth", map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint, strict=False)

        model.eval()

        self.model = model.to(self.device)

        ## dummy input
        dummy = torch.from_numpy(np.zeros([1,3,1080, 1920], dtype = np.float32)).to(self.device)
        self.model(dummy)

        cap = cv2.VideoCapture(vid_path)
        
        self.model = model
        self.cap = cap
        self.ret = True
        self.lr_frame_list = []
        self.hr_frame_list = []
        self.infer_time = []

        self.key = ''

    async def vid_reader(self):
        loop = asyncio.get_event_loop()
        def read_frame():
            ret, frame = self.cap.read()
            frame = np.expand_dims(frame,0).transpose(0,3,1,2)
            frame = torch.from_numpy(frame)
            return ret, frame
                              
        try:
            for _ in range(1000):
                self.ret, frame = await loop.run_in_executor(None, read_frame)
                tic = time.time()
                self.lr_frame_list.append(frame)
                if not self.ret or self.key == ord('q'):
                    break

            self.ret = False

        except asyncio.CancelledError:
            pass

    async def sr_module(self):
        '''
        There are three steps for super resolution
        1. Transport each frame into GPU
        2. Execute the super resolution in model
        3. Transport the result of super resolution back to CPU
                    
        '''

        loop = asyncio.get_event_loop()
        def sr_process(frame):
            frame = frame.to(self.device)   #1
            frame_E = self.model(frame)     #2
            frame_E = frame_E.cpu().numpy() #3
            return frame_E

        try:
            time.sleep(1)
            while (True):
                if len(self.lr_frame_list) > 0:
                    frame_E = await loop.run_in_executor(None, sr_process, self.lr_frame_list[0])
                    del self.lr_frame_list[0]
                    self.hr_frame_list.append(frame_E)
                else:
                    await loop.run_in_executor(None, time.sleep, 1e-3)

                if not(self.ret) and len(self.lr_frame_list) == 0 and len(self.hr_frame_list) == 0:
                    break

                if self.key == ord('q'):
                    break

        except asyncio.CancelledError:
            pass

    async def visualizer(self):
        loop = asyncio.get_event_loop()
        
        try:
            tic = time.time()
            while (True):
                if len(self.hr_frame_list) > 0:
                    frame = self.hr_frame_list.pop(0)
                    cv2.imshow('',frame)
                    self.infer_time.append(time.time()-tic)
                    tic = time.time()
                    
                    self.key = cv2.waitKey(1) & 0xff
                    if self.key == ord('q'):
                        break

                else:
                    await loop.run_in_executor(None, time.sleep, 1e-2)

                if not(self.ret) and len(self.lr_frame_list) == 0 and len(self.hr_frame_list) == 0:
                    break

                if self.key == ord('q'):
                    break
        except asyncio.CancelledError:
            pass

    async def run_asyncio(self):
        futures = [
            asyncio.ensure_future(self.vid_reader()),
            asyncio.ensure_future(self.sr_module()),
            asyncio.ensure_future(self.visualizer())
        ]
        await asyncio.gather(*futures)

    def run(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.run_asyncio())
        loop.close()





with torch.jit.optimized_execution(True):
    start_time = int(round(time.time() * 1000))
    PC = DEMO(args.vid_path)
    PC.run()
    end_time = int(round(time.time() * 1000))
    print("demo_time:", end_time - start_time)
    print(np.mean(PC.infer_time))
