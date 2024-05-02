from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import os
import re
from flask_cors import CORS
import math
import numpy as np

app = Flask(__name__)
CORS(app)

modelCataract1 = YOLO("CataractWeight1/best.pt")
modelCholec1=YOLO("CholecWeight1/best.pt")
modelCataract2=YOLO("CataractWeight2/best.pt")
modelCholec2=YOLO("CholecWeight2/best.pt")

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def phase_sort_key_cataract(phase):
    match = re.search(r'(\d+)', phase)
    if match:
        phase_number = int(match.group())
    else:
        phase_number = float('inf')

    if phase.startswith('Phase7a'):
        phase_number += 0.5
        
    return phase_number


def phase_sort_key_cholec(phase):
    return int(re.search(r'\d+', phase).group())

def remove_smaller_elements(lst, last_value):
    index = 0
    while index < len(lst) and lst[index] < last_value:
        index += 1

    del lst[:index]



def refine_phase_boundaries_cataract(predictions,frame_count,frame_rate):
    last_value=0
    phase_groups = {}
    for phase, second in predictions:
        if phase not in phase_groups:
            phase_groups[phase] = []
        phase_groups[phase].append(second)

    refined_predictions = []


    if 'Phase2' in phase_groups:
        phase2_seconds = phase_groups['Phase2']
        phase2_seconds.sort()
        diff = np.diff(phase2_seconds)
        outliers_index = np.where(diff > 60)[0]
        if len(outliers_index) > 0:
            phase_groups['Phase7a'] = phase2_seconds[outliers_index[0] + 1:]
            phase_groups['Phase2'] = phase2_seconds[:outliers_index[0] + 1]

    for phase in sorted(phase_groups.keys(), key=phase_sort_key_cataract):
        phase_seconds = phase_groups[phase]
        if not phase_seconds:
            continue


        remove_smaller_elements(phase_seconds,last_value)
        
        print(phase_seconds)
        phase_seconds.sort()

        q1, q3 = np.percentile(phase_seconds, [25, 75])

        iqr = q3 - q1

        lower_bound = q1 - 0.4 * iqr
        upper_bound = q3 + 0.4 * iqr

        refined_seconds = [second for second in phase_seconds if lower_bound <= second <= upper_bound]

        last_value=refined_seconds[-1]

        if not refined_seconds:
            continue

        print(refined_seconds)

        refined_predictions.append((phase, refined_seconds[0], refined_seconds[-1]))

    for i in range(len(refined_predictions) - 1):
        current_phase_end = refined_predictions[i][2]
        next_phase_start = refined_predictions[i + 1][1]
        time_gap = (next_phase_start - current_phase_end) / 2

        if time_gap % 1.0 == 0:
            refined_predictions[i] = [refined_predictions[i][0], refined_predictions[i][1], current_phase_end + time_gap-1]
            refined_predictions[i + 1] = [refined_predictions[i + 1][0], next_phase_start - time_gap ,
                                           refined_predictions[i + 1][2]]
        else:
            refined_predictions[i] = [refined_predictions[i][0], refined_predictions[i][1],
                                       math.floor(current_phase_end + time_gap)]
            refined_predictions[i + 1] = [refined_predictions[i + 1][0], math.ceil(next_phase_start - time_gap),
                                           refined_predictions[i + 1][2]]

    refined_predictions[0][1] = 0.0
    refined_predictions[-1][2] = math.ceil(frame_count / frame_rate)

    return refined_predictions


def refine_phase_boundaries_cholec(predictions,frame_count,frame_rate):
    last_value=0
    phase_groups = {}
    for phase, second in predictions:
        if phase not in phase_groups:
            phase_groups[phase] = []
        phase_groups[phase].append(second)

    refined_predictions = []

    for phase in sorted(phase_groups.keys(), key=phase_sort_key_cholec):
        phase_seconds = phase_groups[phase]
        if not phase_seconds:
            continue


        remove_smaller_elements(phase_seconds,last_value)
        
        print(phase_seconds)
        phase_seconds.sort()

        q1, q3 = np.percentile(phase_seconds, [25, 75])

        iqr = q3 - q1

        lower_bound = q1 - 0.4 * iqr
        upper_bound = q3 + 0.4 * iqr

        refined_seconds = [second for second in phase_seconds if lower_bound <= second <= upper_bound]

        last_value=refined_seconds[-1]

        if not refined_seconds:
            continue

        print(refined_seconds)

        refined_predictions.append((phase, refined_seconds[0], refined_seconds[-1]))

    for i in range(len(refined_predictions) - 1):
        current_phase_end = refined_predictions[i][2]
        next_phase_start = refined_predictions[i + 1][1]
        time_gap = (next_phase_start - current_phase_end) / 2

        if time_gap % 1.0 == 0:
            refined_predictions[i] = [refined_predictions[i][0], refined_predictions[i][1], current_phase_end + time_gap-1]
            refined_predictions[i + 1] = [refined_predictions[i + 1][0], next_phase_start - time_gap ,
                                           refined_predictions[i + 1][2]]
        else:
            refined_predictions[i] = [refined_predictions[i][0], refined_predictions[i][1],
                                       math.floor(current_phase_end + time_gap)]
            refined_predictions[i + 1] = [refined_predictions[i + 1][0], math.ceil(next_phase_start - time_gap),
                                           refined_predictions[i + 1][2]]

    refined_predictions[0][1] = 0.0
    refined_predictions[-1][2] = math.ceil(frame_count / frame_rate)

    return refined_predictions

def formatTextCataract(model_predictions):

    phase_tools = {
        "Phase1": ["Bonn Forceps , Primary Knife , Secondary Knife"],
        "Phase2": ["Rycroft Cannula , Visco Cannula"],
        "Phase3": ["Cap Cystotome , Cap Forceps"],
        "Phase4": ["Hydro Cannula"],
        "Phase5": ["Phaco Handpiece"],
        "Phase6": ["A/I Handpiece"],
        "Phase7": ["Rycroft Cannula , Visco Cannula"],
        "Phase7a": ["Hydro Cannula , Visco Cannula"], 
        "Phase8": ["Micromanipulator , Lens Injector"],
        "Phase9": ["Micromanipulator , A/I Handpiece"],
        "Phase10": ["Rycroft Cannula , Visco Cannula"],
    }

    phase_eye_parts = {
        "Phase1": "Cornea, Iris, Cataract Lens , Sclera",  # Replace with actual eye parts for each phase
        "Phase2": "Cornea, Iris, Cataract Lens ,Sclera ",
        "Phase3": "Cornea, Iris, Lens Fragments, Sclera",
        "Phase4": "Cornea, Iris, Lens Fragments",
        "Phase5": "Cornea, Lens Fragments, Sclera",
        "Phase6": "Cornea, Capsule, Sclera",
        "Phase7": "Cornea, Capsule, Sclera",
        "Phase7a": "Cornea, Sclera",
        "Phase8": "Cornea, Artificial Lens, Sclera",
        "Phase9": "Cornea, Artificial Lens, Sclera",
        "Phase10": "Cornea, Artificial Lens, Sclera",
    }

    tools_per_phase = {}
    for phase_code, _, _ in model_predictions:
        tools = phase_tools.get(phase_code, [])
        tools_per_phase.setdefault(phase_code, []).extend(tools)

    template = """{{ PhaseName }} Phase: ({{ StartTime }} seconds - {{ EndTime }} seconds)\n--------------------->Tools Used: {{ Tools }}\n--------------------->Eye parts detected: {{ EyeParts }}\n"""

    phase_names = {
        "Phase1": "Incision",
        "Phase2": "Viscous agent injection",
        "Phase3": "Rhexis",
        "Phase4": "Hydrodissection",
        "Phase5": "Phacoemulsificiation",
        "Phase6": "Irrigation and aspiration",
        "Phase7": "Capsule polishing",
        "Phase7a": "Viscous agent injection",
        "Phase8": "Lens implant setting-up",
        "Phase9": "Viscous agent removal",
        "Phase10": "Tonifying and antibiotics",
    }


    finalTxt=''
    for phase_data in model_predictions:
        phase_code, start_time, end_time = phase_data


        phase_name = phase_names.get(phase_code, phase_code)

        tools = ", ".join(tools_per_phase.get(phase_code, []))

        eye_parts = phase_eye_parts.get(phase_code, "Replace with logic to extract eye parts")
    
        filled_template = template.replace("{{ PhaseName }}", phase_name)
        filled_template = filled_template.replace("{{ StartTime }}", str(int(start_time)))
        filled_template = filled_template.replace("{{ EndTime }}", str(int(end_time)))
        filled_template = filled_template.replace("{{ Tools }}", tools)
        filled_template = filled_template.replace("{{ EyeParts }}", eye_parts)

        finalTxt+=filled_template
    return finalTxt


def formatTextCholec(model_predictions):

    phase_tools = {
        "Phase1": ["Grasper"],
        "Phase2": ["Grasper","Hook","Irrigator"],
        "Phase3": ["Grasper","Clipper","Scissors"],
        "Phase4": ["Grasper","Bipolar"],
        "Phase5": ["Grasper","Specimenbag"],
        "Phase6": ["Grasper","Bipolar","Irrigator"],
        "Phase7": ["Grapser","Bipolar","Irrigator"],
    }

    phase_eye_parts = {
        "Phase1": "Liver, Gallbladder",  # Replace with actual eye parts for each phase
        "Phase2": "Gallbladder",
        "Phase3": "Gallbladder, Fat",
        "Phase4": "Gallbladder",
        "Phase5": "Gallbladder",
        "Phase6": "Liver,Gallbladder",
        "Phase7": "Gallbladder",
    }

    tools_per_phase = {}
    for phase_code, _, _ in model_predictions:
        tools = phase_tools.get(phase_code, [])
        tools_per_phase.setdefault(phase_code, []).extend(tools)

    template = """{{ PhaseName }} Phase: ({{ StartTime }} seconds - {{ EndTime }} seconds )\n--------------------->Tools Used: {{ Tools }}\n--------------------->Organs detected: {{ EyeParts }}\n    """

    phase_names = {
        "Phase1": "Preparation",
        "Phase2": "CalotTriangle Dissection",
        "Phase3": "Clipping Cutting",
        "Phase4": "Gallbladder Dissection",
        "Phase5": "Gallbladder Packaging",
        "Phase6": "Cleaning Coagulation",
        "Phase7": "Gallbladder Retraction",
    }


    finalTxt=''
    for phase_data in model_predictions:
        phase_code, start_time, end_time = phase_data


        phase_name = phase_names.get(phase_code, phase_code)

        tools = ", ".join(tools_per_phase.get(phase_code, []))

        eye_parts = phase_eye_parts.get(phase_code, "Replace with logic to extract eye parts")
    
        filled_template = template.replace("{{ PhaseName }}", phase_name)
        filled_template = filled_template.replace("{{ StartTime }}", str(int(start_time)))
        filled_template = filled_template.replace("{{ EndTime }}", str(int(end_time)))
        filled_template = filled_template.replace("{{ Tools }}", tools)
        filled_template = filled_template.replace("{{ EyeParts }}", eye_parts)

        finalTxt+=filled_template+"\n"
    return finalTxt


def cataract_tools(refined_predictions,video_path):


    phase_tools = {
        "Phase1": ["Knife"],
        "Phase2": ["Rycroft-Cannula"],
        "Phase3": ["Capsulorhexis-Cystotome", "Capsulorhexis-Forceps"],
        "Phase4": ["Hydro-Cannula"],
        "Phase5": ["Phacoemulsification-Handpiece"],
        "Phase6": ["AI-Handpiece"],
        "Phase7": ["Rycroft-Cannula"],
        "Phase7a": ["Hydro-Cannula"],
        "Phase8": ["Micromanipulator", "Lens-Injector"],
        "Phase9": ["Micromanipulator", "AI-Handpiece"],
        "Phase10": ["Rycroft-Cannula"],
    }

    tools_info = {}
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    interval_seconds = 1
    interval_frames = int(frame_rate * interval_seconds)
    frame_count = 0

    width=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)


    prev_knife_coordinates = None
    prev_knife_phase1 = None

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        if frame_count % interval_frames == 0:
            # Calculate current time in seconds
            current_time = frame_count / frame_rate

            # Find the phase for current time
            current_phase = None
            for phase, start_time, end_time in refined_predictions:
                if start_time <= current_time <= end_time:
                    current_phase = phase
                    break

            if current_phase:
                # Reset the variables when a new phase begins
                if current_phase != prev_knife_phase1:
                    prev_knife_coordinates = None
                    prev_knife_phase1 = current_phase

                # Get tools for current phase from phase_tools dictionary
                current_phase_tools = phase_tools[current_phase]

                # Perform inference using the model (assuming `model` is defined elsewhere)
                results = modelCataract2(frame)

                for result in results:
                    prediction = np.array(result.boxes.cls)
                    confidence_score = np.array(result.boxes.conf)

                    for i in range(len(prediction)):
                        if confidence_score[i] > 0.8:
                            tool_name = result.names[prediction[i]]
                            
                            # Check if the detected tool is a cannula and modify its name based on the current phase
                            if tool_name.split("-")[-1] == "Cannula":
                                if current_phase == "Phase2":
                                    tool_name = "Rycroft-Cannula"
                                elif current_phase == "Phase4":
                                    tool_name = "Hydro-Cannula"
                                elif current_phase == "Phase7":
                                    tool_name = "Rycroft-Cannula"
                                elif current_phase == "Phase7a":
                                    tool_name = "Hydro-Cannula"
                                elif current_phase == "Phase10":
                                    tool_name = "Rycroft-Cannula"

                            coordinates = result.boxes.xywh[i][0:2].tolist()

                            if tool_name in phase_tools[current_phase]:
                                if current_phase in tools_info:
                                    if tool_name in tools_info[current_phase]:
                                        tools_info[current_phase][tool_name].append((current_time, coordinates))
                                    else:
                                        tools_info[current_phase][tool_name] = [(current_time, coordinates)]
                                else:
                                    tools_info[current_phase] = {tool_name: [(current_time, coordinates)]}

    cap.release()

    end_string=""
    for phase, tools in tools_info.items():
        print(phase)
        end_string+=f"{phase} Predicted Tools\n"
        for tool, movements in tools.items():
            if movements:
                start_region = get_region(movements[0][1][0], movements[0][1][1],width,height)
                end_region = get_region(movements[-1][1][0], movements[-1][1][1],width,height)
                print(f"{tool} moving from {start_region} to {end_region}")
                end_string+=f"--------> {tool} moving from {start_region} to {end_region}\n\n"
            else:
                print(f"{tool} has no movements.")

    print(end_string)
    return end_string


def cholec_tools(refined_predictions,video_path):
    phase_tools = {
        "Phase1": ["Grasper","L-hook Electrocautery"],
        "Phase2": ["Grasper","L-hook Electrocautery"],
        "Phase3": ["Grasper","L-hook Electrocautery"],
        "Phase4": ["Grasper","L-hook Electrocautery"],
        "Phase5": ["Grasper","L-hook Electrocautery"],
        "Phase6": ["Grasper","L-hook Electrocautery"],
        "Phase7": ["Grapser","L-hook Electrocautery"],
    }


    tools_info = {}

    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    interval_seconds = 10
    interval_frames = int(frame_rate * interval_seconds)
    frame_count = 0

    width=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        if frame_count % interval_frames == 0:
            current_time = frame_count / frame_rate

            # Find the phase for current time
            current_phase_cholec = None
            for phase, start_time, end_time in refined_predictions:
                if start_time <= current_time <= end_time:
                    current_phase_cholec = phase
                    break

            if current_phase_cholec:

                # Get tools for current phase from phase_tools dictionary
                current_phase_cholec_tools = phase_tools[current_phase_cholec]

                # Perform inference using the model (assuming `model` is defined elsewhere)
                results = modelCholec2(frame)

                for result in results:
                    prediction = np.array(result.boxes.cls)
                    confidence_score = np.array(result.boxes.conf)

                    for i in range(len(prediction)):
                        if confidence_score[i] > 0.8:
                            tool_name = result.names[prediction[i]]
                            


                            coordinates = result.boxes.xywh[i][0:2].tolist()

                            if tool_name in phase_tools[current_phase_cholec]:
                                if current_phase_cholec in tools_info:
                                    if tool_name in tools_info[current_phase_cholec]:
                                        tools_info[current_phase_cholec][tool_name].append((current_time, coordinates))
                                    else:
                                        tools_info[current_phase_cholec][tool_name] = [(current_time, coordinates)]
                                else:
                                    tools_info[current_phase_cholec] = {tool_name: [(current_time, coordinates)]}

    cap.release()


    end_string=""
    for phase, tools in tools_info.items():
        print(phase)
        end_string+=f"{phase} Predicted Tools\n"
        for tool, movements in tools.items():
            if movements:
                start_region = get_region(movements[0][1][0], movements[0][1][1],width,height)
                end_region = get_region(movements[-1][1][0], movements[-1][1][1],width,height)
                print(f"{tool} moving from {start_region} to {end_region}")
                end_string+=f"--------> {tool} moving from {start_region} to {end_region}\n\n"
            else:
                print(f"{tool} has no movements.")

    print(end_string)
    return end_string

# Define function to get region based on coordinates
def get_region(x, y,width,height):
    if x < width / 3:
        if y < height / 3:
            return "Top-Left"
        elif y < 2 * height / 3:
            return "Mid-Left"
        else:
            return "Bottom-Left"
    elif x < 2 * width / 3:
        if y < height / 3:
            return "Top-Center"
        elif y < 2 * height / 3:
            return "Mid-Center"
        else:
            return "Bottom-Center"
    else:
        if y < height / 3:
            return "Top-Right"
        elif y < 2 * height / 3:
            return "Mid-Right"
        else:
            return "Bottom-Right"





def process_video_cataract(video_path):

    # return [['Phase1', 0.0, 22.0],
    # ['Phase2', 23.0, 28.0],
    # ['Phase3', 29.0, 56],
    # ['Phase4', 57, 84],
    # ['Phase5', 85, 146],
    # ['Phase6', 147, 196.0],
    # ['Phase7', 197.0, 203.0],
    # ['Phase7a', 204.0, 217.0],
    # ['Phase8', 218.0, 239],
    # ['Phase9', 240, 283.0],
    # ['Phase10', 284.0, 289]]

    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    interval_seconds = 1
    interval_frames = int(frame_rate * interval_seconds)

    frame_count = 0
    output1 = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1
        
        if frame_count % interval_frames == 0:
            result = modelCataract1(frame)
            temp = result[0]
            predicted_phases = temp.names[temp.probs.top1]
            if temp.probs.top1conf>0.9:
                output1.append((predicted_phases,frame_count/frame_rate))

    cap.release()

    refined_predictions = refine_phase_boundaries_cataract(output1,frame_count,frame_rate)

    return refined_predictions

def process_video_cholec(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    interval_seconds = 10
    interval_frames = int(frame_rate * interval_seconds)

    frame_count = 0
    output1 = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1
        
        if frame_count % interval_frames == 0:
            result = modelCholec1(frame)
            temp = result[0]
            predicted_phases = temp.names[temp.probs.top1]
            if temp.probs.top1conf>0.9:
                output1.append((predicted_phases,frame_count/frame_rate))

    cap.release()

    refined_predictions = refine_phase_boundaries_cholec(output1,frame_count,frame_rate)

    return refined_predictions



@app.route('/process_video_cataract', methods=['POST'])
def process_video_route_cataract():
    if 'video' not in request.files:
        return jsonify({'error': 'No file part'})

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'})

    video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
    video_file.save(video_path)

    result = process_video_cataract(video_path)
    print(result)
    result1=formatTextCataract(result)
    result2=cataract_tools(result,video_path)

    os.remove(video_path)  # Remove the video file after processing
    return jsonify({"message":result1,"message2":result2})


@app.route('/process_video_cholec', methods=['POST'])
def process_video_route_cholec():
    if 'video' not in request.files:
        return jsonify({'error': 'No file part'})

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'})

    video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
    video_file.save(video_path)

    result = process_video_cholec(video_path)
    result1=formatTextCholec(result)
    result2=cholec_tools(result,video_path)

    os.remove(video_path)  # Remove the video file after processing
    return jsonify({"message":result1,"message2":result2})

if __name__ == '__main__':
    app.run(debug=True,port=8000)
