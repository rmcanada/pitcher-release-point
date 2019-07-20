import os
import json
import urllib.request
import subprocess
import argparse
import shutil


parser = argparse.ArgumentParser()
parser.add_argument('--guid', help='guid from json of video')

json_files = [pos_json for pos_json in os.listdir('.') if pos_json.endswith('.json')]


def download_video(url):
    try:
        ret1 = subprocess.call(['wget',"-P", "./",url,])
        if ret1 > 0:
            raise Exception('could not wget')
    except Exception as e:
        print (str(e))
        raise e

def get_video_url(pitch_dict):
    # check to see that the data pertains to a pitch and not a pickoff
    if (pitch_dict['isPickoff'] and not pitch_dict['isPitch']):
        print("Not pitch, pick off")
        return "Pickoff"
    
    # subset info on playback types to iterate through
    video_playbacks = pitch_dict['video']['playbackGroups']

    # loop through playback types looking for centrefield camera
    for i in range(len(video_playbacks)):
        if video_playbacks[i]['mediaSourceType'] == "CENTERFIELD":
            max_bit = 0
            max_bit_index = None
            # create subset of playback data pertaining to current data
            cf = video_playbacks[i]
            for iter, j in enumerate(cf['playbackRenditions']):
                # if bitrate is higher than current max bitrate, pick
                bitrate = int(j['bitrate'].strip("K"))
                if bitrate > max_bit:
                    max_bit = bitrate
                    max_bit_index = iter
                    # print(max_bit_index)
    video_url = cf['playbackRenditions'][max_bit_index]['playbackUrl']
    return video_url


def build_dict(seq, key):
    return dict((d[key], dict(d, index=index)) for (index, d) in enumerate(seq))


def json_data(guid):

    for j in json_files:
        with open(j, 'r') as myfile:
            data=myfile.read()
            if guid in data:
                # print(j)
                obj = json.loads(data)
                # print("got obj")
                return obj
        # else:
    print("Cannot find GUID, please enter GUID from a JSON file in current dir")
    return

def clear_images():
    folder = './images/'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

def main():
    args = parser.parse_args()
    guid = args.guid

    # search in json files for the guid
    obj = json_data(guid)

    if obj is not None:
        info = build_dict(obj, key="guid")
        # pickoff attempt example
        # pitch_dict = info.get("09812143-51f6-4dd9-9789-d72c53354980")
        pitch_dict = info.get(guid)
        url = get_video_url(pitch_dict)
        vid = url.split('https://sporty-clips.mlb.com/')[1]
        vid_img = vid.split("-")[0]
        if url is not None:
            # delete old videos
            for item in os.listdir("."):
                if item.endswith(".mp4"):
                    os.remove(item)
            download_video(url)
            print("removing old images...")
            clear_images()
            subprocess.call(['ffmpeg','-i', vid, "-q:v","1",'images/{}_%04d.jpg'.format(vid_img,"")])
        else: print("no url found")


if __name__ == '__main__':
    main()
        
        # ret = subprocess.call(['ffmpeg','-i', video_src, '-c' , 'copy', '-t', clip_duration, 
        #     '-vcodec',  'libx264', '{}/{}_{}.mp4'.format(event_type,os.path.splitext(video_src)[0],"{:02d}".format(clip_num))])
        # if ret > 0:
        #     raise Exception('ffmpeg could not split the video to clips')

# print('Beginning file download... {}'.format(url))
# urllib.request.urlretrieve("{}".format(url), headers={'User-Agent': 'Mozilla/5.0'})
# data = urllib.request.urlretrieve("{}".format(url))


# req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
# webpage = urlopen(req).read()



# urllib.request.urlretrieve(link, 'video_name.mp4') 
# urllib.request.urlretrieve(link)  

# # print the keys and values
# for key in obj:
#     # loop through each entry in the json file, which is a recorded pitch (key is a pitch)
#     # get the video list from each pitch which is called playbackGroup, returns list
#     vid_playback = key['video']['playbackGroups']
#     for i in range(len(vid_playback)):
#         # iterate through list and get centerfield playbackrenditions
#         # extract centerfield camera and max bitrate
#         if vid_playback[i]['mediaSourceType'] == "CENTERFIELD":
#             max_bit = 0
#             max_bit_index = None
#             cf = vid_playback[i]
#             for iter, j in enumerate(cf['playbackRenditions']):
#                 # print(iter)
#                 # print(j)
#                 bitrate = int(j['bitrate'].strip("K"))
#                 if bitrate > max_bit:
#                     max_bit = bitrate
#                     max_bit_index = iter
#                     # print(max_bit_index)
#             print(cf['playbackRenditions'][max_bit_index]['playbackUrl'])