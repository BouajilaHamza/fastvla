import os
import json
import re
import pickle
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download, list_repo_files

print("Starting Arabic Unitree G1 Dataset preparation and translation...")

SOURCE_REPO = "openhe/g1-retargeted-motions"
TARGET_REPO = "hamzabouajila/ar-unitree-g1-motions"
CACHE_DIR = "/tmp/unitree_g1_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

api = HfApi()

# Fetch file list
all_files = list_repo_files(repo_id=SOURCE_REPO, repo_type="dataset")
pkl_files = sorted([f for f in all_files if f.endswith(".pkl")])
print(f"Discovered {len(pkl_files)} motion pickle files in {SOURCE_REPO}.")

def generate_motion_metadata(filename):
    cat_dir = filename.split('/')[0]
    base = filename.split('/')[1]
    base_lower = base.lower()
    
    # 1. Kung Fu / Martial Arts
    if cat_dir == 'kungfu_retargeted':
        if 'Bruce_Lee' in base:
            return {
                "instruction_ar": "اتخذ وضعية بروس لي القتالية الكلاسيكية مع الاستعداد للدفاع والهجوم",
                "instruction_en": "Adopt the classic Bruce Lee martial arts stance ready for offense and defense.",
                "category": "martial_arts",
                "variations_ar": ["قف في وضعية بروس لي القتالية", "استعد للقتال بوضعية بروس لي الشهيرة", "اتخذ وقفة قتالية رشيقة بأسلوب بروس لي"]
            }
        elif 'Charleston' in base:
            return {
                "instruction_ar": "أدِّ رقصة تشارلستون الإيقاعية الكلاسيكية بحركات أقدام وركبتين متناسقة",
                "instruction_en": "Perform the classic energetic Charleston dance with rhythmic footwork.",
                "category": "dance",
                "variations_ar": ["ارقص تشارلستون بخطوات سريعة", "قم بأداء رقصة تشارلستون التعبيرية", "تحرك بإيقاع رقصة تشارلستون"]
            }
        elif 'Hooks_punch' in base:
            return {
                "instruction_ar": "سدد لكمات خطافية قوية ومتتالية بالذراعين مع دوران الجذع",
                "instruction_en": "Throw powerful alternating hook punches with torso rotation.",
                "category": "martial_arts",
                "variations_ar": ["سدد لكمة خطافية خاطفة", "اضرب بلكمات خطافية متتالية", "نفذ هجوم اللكمات الخطافية في الفنون القتالية"]
            }
        elif 'Horse-stance_pose' in base:
            return {
                "instruction_ar": "اتخذ وضعية ركوب الخيل القتالية (مابو) مع ثبات القدمين وانحناء الركبتين",
                "instruction_en": "Adopt the solid martial arts horse stance (Ma Bu) with bent knees.",
                "category": "martial_arts",
                "variations_ar": ["اثبت في وضعية ركوب الخيل", "اتخذ وقفة مابو القتالية الثابتة", "اخفض مركز الجاذبية في وقفة ركوب الخيل"]
            }
        elif 'Horse-stance_punch' in base:
            return {
                "instruction_ar": "سدد لكمة أمامية مستقيمة وقوية انطلاقاً من وضعية ركوب الخيل الثابتة",
                "instruction_en": "Execute a powerful straight punch from the horse stance pose.",
                "category": "martial_arts",
                "variations_ar": ["اضرب بلكمة مستقيمة من وضعية ركوب الخيل", "سدد لكمة كونغ فو من وقفة الخيل", "نفذ لكمة أمامية من وضعية مابو"]
            }
        elif 'Roundhouse_kick' in base:
            return {
                "instruction_ar": "نفذ ركلة دائرية سريعة وقوية بالساق مع الارتكاز والدوران",
                "instruction_en": "Execute a high-velocity roundhouse kick with pivot and hip drive.",
                "category": "martial_arts",
                "variations_ar": ["اركل ركلة دائرية قوية", "سدد ركلة راوند هاوس هوائية", "نفذ ركلة قتالية دائرية بالساق"]
            }
        elif 'Side_kick' in base:
            return {
                "instruction_ar": "سدد ركلة جانبية مباشرة وقوية مع فرد الساق بالكامل",
                "instruction_en": "Deliver a direct lateral side kick with full leg extension.",
                "category": "martial_arts",
                "variations_ar": ["اركل ركلة جانبية مستقيمة", "سدد ركلة دفع جانبية", "نفذ ركلة جانبية سريعة نحو الهدف"]
            }
        elif 'merged_motion' in base:
            return {
                "instruction_ar": "نفذ تسلسلاً قتالياً شاملاً يجمع بين اللكمات والركلات والوقفات المتنوعة",
                "instruction_en": "Perform a comprehensive martial arts combat sequence combining punches, kicks, and stances.",
                "category": "martial_arts",
                "variations_ar": ["أدِّ تسلسل قتالي مدمج للكونغ فو", "تحرك بسلسلة حركات فنون قتالية متتالية", "استعرض مهارات قتالية متعددة"]
            }

    # 2. Dance DB
    if cat_dir == 'dance_db_retargeted':
        if 'Hasaposerviko' in base:
            return {
                "instruction_ar": "أدِّ رقصة هاسابوسيرفيكو اليونانية السريعة بخطوات إيقاعية رشيقة",
                "instruction_en": "Perform the fast-paced traditional Greek Hasaposerviko dance with agile footwork.",
                "category": "dance",
                "variations_ar": ["ارقص هاسابوسيرفيكو بخطوات سريعة متتالية", "قم بأداء رقصة هاسابوسيرفيكو الفولكلورية", "تحرك بإيقاع الرقصة اليونانية السريعة"]
            }
        elif 'Sta_tria_Syrtos' in base:
            return {
                "instruction_ar": "أدِّ رقصة سيرتوس الشعبية اليونانية بثلاث خطوات تقليدية متزنة",
                "instruction_en": "Perform the traditional Greek Syrtos dance in three rhythmic steps.",
                "category": "dance",
                "variations_ar": ["ارقص رقصة سيرتوس التقليدية", "أدِّ خطوات سيرتوس الثلاث الفولكلورية", "تحرك بأسلوب رقصة سيرتوس اليونانية"]
            }
        elif 'Tatsia' in base:
            return {
                "instruction_ar": "أدِّ رقصة التاتسيا القبرصية التراثية بحركات استعراضية متوازنة",
                "instruction_en": "Perform the traditional Cypriot Tatsia folk dance with balanced movements.",
                "category": "dance",
                "variations_ar": ["ارقص رقصة التاتسيا التراثية", "أدِّ رقصة تاتسيا الفولكلورية", "تحرك باستعراض رقصة التاتسيا"]
            }
        elif 'Tsamiko' in base:
            return {
                "instruction_ar": "أدِّ رقصة تساميكو اليونانية الحماسية بخطوات شجاعة ووثبات مدروسة",
                "instruction_en": "Perform the heroic Greek Tsamiko dance with bold leaps and disciplined steps.",
                "category": "dance",
                "variations_ar": ["ارقص تساميكو بخطوات حماسية", "أدِّ رقصة تساميكو البطولية", "تحرك بأسلوب رقصة تساميكو الشعبية"]
            }
        elif 'Zeibekiko' in base:
            return {
                "instruction_ar": "أدِّ رقصة زيبكيكو الفردية بإيقاع بطيء متزن ودورانات معبرة",
                "instruction_en": "Perform the expressive solo Zeibekiko dance with deliberate turns and pauses.",
                "category": "dance",
                "variations_ar": ["ارقص زيبكيكو الفردية بتمهل وتركيز", "أدِّ رقصة زيبكيكو الإيقاعية", "تحرك بتعبير درامي في رقصة زيبكيكو"]
            }
        elif 'RandB' in base:
            return {
                "instruction_ar": "أدِّ حركات رقصة الآر أند بي المعاصرة بانسيابية وإيقاع حضري حديث",
                "instruction_en": "Perform contemporary R&B dance moves with smooth urban groove and musicality.",
                "category": "dance",
                "variations_ar": ["ارقص آر أند بي بحركات جسد مرنة", "أدِّ رقصة R&B عصرية", "تمايل بإيقاع موسيقى الآر أند بي"]
            }
        elif 'Capoeira' in base:
            return {
                "instruction_ar": "أدِّ حركات رقصة الكابويرا البرازيلية القتالية مع التمايل المستمر (الجينغا)",
                "instruction_en": "Perform Brazilian Capoeira martial dance movements with continuous Ginga flow.",
                "category": "dance",
                "variations_ar": ["تحرك بأسلوب الكابويرا القتالية", "أدِّ رقصة كابويرا بحركات دائرية وتمايل مستمر", "مارس حركات الكابويرا البرازيلية"]
            }
        elif 'Afraid' in base:
            return {
                "instruction_ar": "تحرك بحركات تعبيرية تدل على الخوف والتردد مع التراجع وحماية الجسد",
                "instruction_en": "Express fear through hesitant steps, protective posture, and defensive movements.",
                "category": "expressive",
                "variations_ar": ["عبر عن مشاعر الخوف بحركات متراجعة ومترددة", "اتخذ وقفة حذرة وخائفة", "تحرك بوجل وتردد لحماية نفسك"]
            }
        elif 'Angry' in base:
            return {
                "instruction_ar": "تحرك بحركات تعبيرية غاضبة مع حركات ذراعين حادة ومشي مشحون بالطاقة",
                "instruction_en": "Express anger through tense posture, sharp arm motions, and forceful stomps.",
                "category": "expressive",
                "variations_ar": ["عبر عن الغضب بحركات قوية وحادة", "امشِ بانفعال وغضب ظاهر", "أظهر تعبيرات الغضب بحركات جسد حاسمة"]
            }
        elif 'Excited' in base:
            return {
                "instruction_ar": "تحرك بحركات حيوية تعبر عن الحماس والفرح مع قفزات خفيفة وتلويح",
                "instruction_en": "Express excitement with energetic bounces, open gestures, and enthusiastic motion.",
                "category": "expressive",
                "variations_ar": ["أظهر الحماس بحركات نشيطة وقوية", "تحرك ببهجة وتلويح حماسي", "عبر عن الإثارة والحماس بالقفز الخفيف"]
            }
        elif 'Happy' in base:
            return {
                "instruction_ar": "تحرك بسعادة وابتهاج مع مشي مرن وخفيف وإيماءات إيجابية",
                "instruction_en": "Express happiness with light steps, buoyant posture, and cheerful gestures.",
                "category": "expressive",
                "variations_ar": ["امشِ بسعادة وانشراح", "تحرك بحركات تعبيرية سعيدة ومبهجة", "أظهر الفرح بخطوات خفيفة ومرحة"]
            }
        elif 'Relaxed' in base:
            return {
                "instruction_ar": "تحرك باسترخاء وهدوء تام مع انسيابية في الذراعين وتمايل سلس",
                "instruction_en": "Move in a relaxed and calm manner with loose limbs and gentle swaying.",
                "category": "expressive",
                "variations_ar": ["امشِ بهدوء واسترخاء", "تحرك بحركات هادئة غير مشدودة", "عبر عن الاسترخاء بتمهل وسلاسة"]
            }
        elif 'Sad' in base:
            return {
                "instruction_ar": "تحرك بحركات حزينة منكس الرأس مع خطوات بطيئة وذراعين هابطتين",
                "instruction_en": "Express sadness with dropped head, slumping shoulders, and heavy sluggish steps.",
                "category": "expressive",
                "variations_ar": ["امشِ ببطء وحزن مع ثني الرأس للأسفل", "عبر عن الحزن بحركات ثقيلة ومنخفضة", "تحرك بخطوات واهنة تعبر عن الكآبة"]
            }

    # 3. LaFAN1
    if cat_dir == 'lafan1_retargeted':
        if 'dance1' in base:
            return {
                "instruction_ar": "أدِّ رقصة إيقاعية حركية مع خطوات متزامنة وتلويح بالذراعين",
                "instruction_en": "Perform a rhythmic dance routine with synchronized steps and arm movements.",
                "category": "dance",
                "variations_ar": ["ارقص رقصة إيقاعية نشيطة", "أدِّ تسلسل رقص حركي متناسق", "تحرك بإيقاع راقص وتلويح اليدين"]
            }
        elif 'dance2' in base:
            return {
                "instruction_ar": "أدِّ حركات رقص تعبيرية مع دوران الجذع وانسيابية الخطوات",
                "instruction_en": "Perform expressive fluid dance choreography with torso rotations and steps.",
                "category": "dance",
                "variations_ar": ["ارقص بحركات جسد انسيابية", "أدِّ حركات رقص تعبيري متواصل", "تحرك بتناغم راقص وسلس"]
            }
        elif 'fallAndGetUp' in base:
            return {
                "instruction_ar": "سقوط متحكم به على الأرض ثم النهوض بحذر واستعادة وضعية الوقوف",
                "instruction_en": "Controlled fall to the ground followed by careful recovery and standing up.",
                "category": "whole_body_gestures",
                "variations_ar": ["اسقط على الأرض ثم انهض مجدداً", "قم بعد السقوط واستعد الوقوف", "نفذ حركة السقوط والنهوض الذاتي"]
            }
        elif 'fight1' in base or 'fightAndSports' in base:
            return {
                "instruction_ar": "أدِّ حركات قتالية ودفاعية ديناميكية تجمع بين المناورة والضربات السريعة",
                "instruction_en": "Perform dynamic fighting and sports maneuvers with evasions and strikes.",
                "category": "martial_arts",
                "variations_ar": ["تحرك بحركات قتالية ورياضية سريعة", "ناور وسدد ضربات دفاعية وهجومية", "مارس حركات قتال وتفادٍ رياضية"]
            }
        elif 'jumps' in base:
            return {
                "instruction_ar": "اقفز قفزة عمودية للأعلى بكلتا الساقين مع الهبوط المتوازن والمرن",
                "instruction_en": "Execute vertical jumps with both legs and absorb landing impact smoothly.",
                "category": "locomotion",
                "variations_ar": ["اقفز للأعلى واهبط بثبات", "نفذ قفزة رياضية متوازنة", "اقفز بكلتا القدمين مع امتصاص الصدمة"]
            }
        elif 'sprint' in base:
            return {
                "instruction_ar": "انطلق في ركض سريع للأمام (سبرنت) بأقصى قوة وتسارع",
                "instruction_en": "Sprint forward with maximum speed, drive, and high leg cadence.",
                "category": "locomotion",
                "variations_ar": ["اركض بأقصى سرعة ممكنة للأمام", "انطلق في عدو سريع ومكثف", "تسارع في سبرنت قوي ومستقيم"]
            }
        elif 'run' in base:
            return {
                "instruction_ar": "اركض للأمام بخطوات سريعة ومنتظمة مع تأرجح الذراعين",
                "instruction_en": "Run forward with a steady, athletic stride and coordinated arm swing.",
                "category": "locomotion",
                "variations_ar": ["اركض للأمام بإيقاع ثابت", "اجرِ بخطوات واسعة وسريعة", "تابع الجري المستقيم للأمام"]
            }
        elif 'walk' in base:
            return {
                "instruction_ar": "امشِ للأمام بخطوات طبيعية منتظمة وثابتة",
                "instruction_en": "Walk forward with a natural, steady, and balanced gait.",
                "category": "locomotion",
                "variations_ar": ["امشِ للأمام بوتيرة طبيعية", "سر بخطوات متزنة ومستقيمة", "تقدم مشياً للأمام بثبات"]
            }

    # 4. ACCAD
    if 'pick_up_box' in base_lower:
        if 'walk_to' in base_lower:
            return {
                "instruction_ar": "امشِ نحو الصندوق ثم انحنِ والتقطه من الأرض بيديك",
                "instruction_en": "Walk towards the box, crouch down, and pick it up from the ground.",
                "category": "manipulation",
                "variations_ar": ["تقدم نحو الصندوق والتقطه", "امشِ للأمام وانحنِ لحمل الصندوق", "سر نحو الصندوق وارفعه بكلتا اليدين"]
            }
        elif 'run_to' in base_lower:
            return {
                "instruction_ar": "اركض نحو الصندوق ثم توقف والتقطه من الأرض بحذر",
                "instruction_en": "Run towards the box, stop, and pick it up carefully.",
                "category": "manipulation",
                "variations_ar": ["اجرِ نحو الصندوق وارفعه", "اركض وتوقف عند الصندوق لالتقاطه", "أسرع نحو الصندوق واحمله من الأرض"]
            }
        else:
            return {
                "instruction_ar": "انحنِ لأسفل والتقط الصندوق من الأرض بكلتا اليدين ثم قف",
                "instruction_en": "Crouch down, grasp the box from the floor with both hands, and stand up.",
                "category": "manipulation",
                "variations_ar": ["التقط الصندوق من الأرض وقف", "احمل الصندوق بيديك من الأرضية", "انحنِ وارفع الصندوق بكلتا يديك"]
            }
            
    if 'lift_box' in base_lower or 'box_lift' in base_lower:
        return {
            "instruction_ar": "ارفع الصندوق بكلتا اليدين للأعلى بثبات مع شد الظهر والذراعين",
            "instruction_en": "Lift the box upward with both hands while maintaining steady posture.",
            "category": "manipulation",
            "variations_ar": ["ارفع الصندوق للأعلى بكلتا اليدين", "احمل الصندوق وارفع مستواه بثبات", "قم برفع الصندوق للأعلى"]
        }
        
    if 'put_down_box' in base_lower:
        return {
            "instruction_ar": "انحنِ وضع الصندوق على الأرض بعناية ثم واصل المشي",
            "instruction_en": "Lower down and carefully place the box on the ground, then resume walking.",
            "category": "manipulation",
            "variations_ar": ["ضع الصندوق على الأرض وتابع المشي", "انزل الصندوق للأرض بحذر وواصل السير", "اترك الصندوق على الأرض وتقدم مشياً"]
        }

    if 'walk_with_box' in base_lower:
        return {
            "instruction_ar": "امشِ للأمام بثبات أثناء حمل الصندوق بكلتا اليدين أمام الجسد",
            "instruction_en": "Walk forward steadily while holding the box in front of the body with both hands.",
            "category": "manipulation",
            "variations_ar": ["سر حاملاً الصندوق بين يديك", "امشِ للأمام والصندوق محمول بيديك", "تقدم مشياً مع الإمساك بالصندوق أمامك"]
        }

    if 'walk' in base_lower and 'turn' in base_lower:
        angle_match = re.search(r'\(?(\d+)\)?', base_lower)
        angle = angle_match.group(1) if angle_match else ""
        if 'left' in base_lower:
            deg_str = f" بزاوية {angle} درجة" if angle else ""
            return {
                "instruction_ar": f"امشِ للأمام ثم انعطف إلى اليسار{deg_str} وواصل المشي",
                "instruction_en": f"Walk forward and turn left {angle} degrees, then continue walking.",
                "category": "locomotion",
                "variations_ar": [f"انعطف يساراً{deg_str} أثناء المشي", f"سر للأمام ودر نحو اليسار{deg_str}", f"امشِ وانحرف لليسار{deg_str}"]
            }
        elif 'right' in base_lower:
            deg_str = f" بزاوية {angle} درجة" if angle else ""
            return {
                "instruction_ar": f"امشِ للأمام ثم انعطف إلى اليمين{deg_str} وواصل المشي",
                "instruction_en": f"Walk forward and turn right {angle} degrees, then continue walking.",
                "category": "locomotion",
                "variations_ar": [f"انعطف يميناً{deg_str} أثناء المشي", f"سر للأمام ودر نحو اليمين{deg_str}", f"امشِ وانحرف لليمين{deg_str}"]
            }
        elif 'around' in base_lower or 'change' in base_lower:
            return {
                "instruction_ar": "امشِ للأمام ثم استدر في الاتجاه المعاكس وواصل السير",
                "instruction_en": "Walk forward, turn around to reverse direction, and continue walking.",
                "category": "locomotion",
                "variations_ar": ["استدر 180 درجة أثناء المشي", "امشِ ثم انعطف للخلف وتابع السير", "غير اتجاه مشيك بالالتفاف الكامل"]
            }

    if 'run' in base_lower and 'turn' in base_lower:
        angle_match = re.search(r'\(?(\d+)\)?', base_lower)
        angle = angle_match.group(1) if angle_match else ""
        if 'left' in base_lower:
            deg_str = f" بزاوية {angle} درجة" if angle else ""
            return {
                "instruction_ar": f"اركض للأمام ثم انعطف إلى اليسار{deg_str} وواصل الركض",
                "instruction_en": f"Run forward and turn left {angle} degrees, then continue running.",
                "category": "locomotion",
                "variations_ar": [f"اجرِ وانعطف يساراً{deg_str}", f"اركض مع الدوران لليسار{deg_str}", f"غير مسار ركضك نحو اليسار{deg_str}"]
            }
        elif 'right' in base_lower:
            deg_str = f" بزاوية {angle} درجة" if angle else ""
            return {
                "instruction_ar": f"اركض للأمام ثم انعطف إلى اليمين{deg_str} وواصل الركض",
                "instruction_en": f"Run forward and turn right {angle} degrees, then continue running.",
                "category": "locomotion",
                "variations_ar": [f"اجرِ وانعطف يميناً{deg_str}", f"اركض مع الدوران لليمين{deg_str}", f"غير مسار ركضك نحو اليمين{deg_str}"]
            }
        elif 'around' in base_lower or 'change' in base_lower:
            return {
                "instruction_ar": "اركض للأمام ثم استدر في الاتجاه المعاكس وواصل الجري",
                "instruction_en": "Run forward, pivot around to reverse direction, and continue running.",
                "category": "locomotion",
                "variations_ar": ["استدر بالكامل أثناء الجري للخلف", "اركض والتف 180 درجة وتابع الركض", "غير اتجاه ركضك بسرعة"]
            }

    if 'side_step' in base_lower or 'step_left' in base_lower or 'step_right' in base_lower:
        direction = "اليسار" if "left" in base_lower else "اليمين"
        dir_en = "left" if "left" in base_lower else "right"
        is_quick = "quick" in base_lower
        quick_ar = "سريعة " if is_quick else ""
        return {
            "instruction_ar": f"تحرك بخطوات جانبية {quick_ar}نحو {direction}",
            "instruction_en": f"Take {'quick ' if is_quick else ''}side steps towards the {dir_en}.",
            "category": "locomotion",
            "variations_ar": [f"اخطُ خطوة جانبية باتجاه {direction}", f"تحرك جانباً نحو {direction}", f"سر بخطى جانبية إلى {direction}"]
        }

    if 'backwards' in base_lower or 'walk_back' in base_lower:
        if 'crawl' in base_lower:
            return {
                "instruction_ar": "ازحف على أطرافك الأربعة إلى الخلف ببطء",
                "instruction_en": "Crawl backwards on all fours steadily.",
                "category": "locomotion",
                "variations_ar": ["ازحف للخلف على الأرض", "تحرك زحفاً إلى الوراء", "تراجع بالزحف للخلف"]
            }
        elif 'run' in base_lower:
            return {
                "instruction_ar": "اركض إلى الخلف بحذر مع الحفاظ على التوازن",
                "instruction_en": "Run backwards carefully while maintaining balance.",
                "category": "locomotion",
                "variations_ar": ["اجرِ للخلف بحذر", "تراجع بالركض السريع إلى الوراء", "اركض إلى الوراء بثبات"]
            }
        else:
            return {
                "instruction_ar": "امشِ إلى الخلف بخطوات منتظمة ومتزنة",
                "instruction_en": "Walk backwards with steady, balanced steps.",
                "category": "locomotion",
                "variations_ar": ["تراجع بالمشي إلى الوراء", "امشِ للخلف بتمهل واتزان", "سر إلى الخلف بخطوات ثابتة"]
            }

    if 'crawl' in base_lower:
        if 'military' in base_lower:
            dir_str = "للخلف" if "backwards" in base_lower else "للأمام"
            dir_en = "backwards" if "backwards" in base_lower else "forward"
            return {
                "instruction_ar": f"ازحف زحفاً عسكرياً منخفضاً وملاصقاً للأرض {dir_str}",
                "instruction_en": f"Perform a low military belly crawl {dir_en}.",
                "category": "locomotion",
                "variations_ar": [f"ازحف على البطن بأسلوب عسكري {dir_str}", f"تقدم زحفاً عسكرياً منخفضاً {dir_str}", f"ازحف بمحاذاة الأرض {dir_str}"]
            }
        else:
            dir_str = "للخلف" if "backwards" in base_lower else "للأمام"
            dir_en = "backwards" if "backwards" in base_lower else "forward"
            return {
                "instruction_ar": f"ازحف على اليدين والركبتين {dir_str}",
                "instruction_en": f"Crawl on hands and knees {dir_en}.",
                "category": "locomotion",
                "variations_ar": [f"تقدم زحفاً على الأطراف الأربعة {dir_str}", f"ازحف {dir_str} بثبات", f"سر زحفاً على ركبتيك ويديك {dir_str}"]
            }

    if 'lie' in base_lower:
        if 'crouch_to_lie' in base_lower or 'crourch_to_lie' in base_lower:
            return {
                "instruction_ar": "انتقل من وضعية القرفصاء/الانحناء إلى الاستلقاء الكامل على الأرض",
                "instruction_en": "Transition from a crouching posture down to lying flat on the floor.",
                "category": "whole_body_gestures",
                "variations_ar": ["استلقِ على الأرض بعد الانحناء", "انتقل من القرفصاء إلى الاستلقاء التام", "انبطح أرضاً من وضع القرفصاء"]
            }
        elif 'lie_to_crouch' in base_lower or 'lie_down_to_crouch' in base_lower:
            return {
                "instruction_ar": "انهض من الاستلقاء على الأرض إلى وضعية القرفصاء/الانحناء",
                "instruction_en": "Push up from lying on the floor into a crouch position.",
                "category": "whole_body_gestures",
                "variations_ar": ["انهض من الاستلقاء إلى القرفصاء", "ارفع جسدك من الأرض واقرفص", "تحول من وضعية النوم إلى القرفصاء"]
            }
        else:
            return {
                "instruction_ar": "استلقِ ممدداً على الأرض بالكامل في وضعية الراحة",
                "instruction_en": "Lie down flat on the floor in a relaxed resting posture.",
                "category": "whole_body_gestures",
                "variations_ar": ["تمدد على الأرضية بالكامل", "انبطح واستلقِ أرضاً", "خذ وضعية الاستلقاء على الأرض"]
            }

    if 'crouch' in base_lower:
        if 'walk_to_crouch' in base_lower:
            return {
                "instruction_ar": "امشِ للأمام ثم اخفض جسدك واتخذ وضعية القرفصاء",
                "instruction_en": "Walk forward and then lower the body into a crouch.",
                "category": "locomotion",
                "variations_ar": ["سر للأمام ثم اقرفص", "امشِ وانحنِ في وقفة قرفصاء", "تقدم ثم اخفض قامتك للقرفصاء"]
            }
        elif 'crouch_to_walk' in base_lower:
            return {
                "instruction_ar": "انهض من وضعية القرفصاء وابدأ بالمشي للأمام",
                "instruction_en": "Rise from a crouch posture and begin walking forward.",
                "category": "locomotion",
                "variations_ar": ["قم من القرفصاء وامشِ للأمام", "انتصب وابدأ السير للأمام", "انهض من انحنائك وتقدم مشياً"]
            }
        elif 'run_to_crouch' in base_lower:
            return {
                "instruction_ar": "اركض للأمام ثم توقف بسرعة وانحنِ في وضعية القرفصاء",
                "instruction_en": "Run forward, brake to a stop, and drop down into a crouch.",
                "category": "locomotion",
                "variations_ar": ["اجرِ وتوقف في وقفة قرفصاء", "اركض واخفض جسدك للقرفصاء سريعاً", "اهبط في قرفصاء بعد الركض"]
            }
        elif 'crouch_to_run' in base_lower:
            return {
                "instruction_ar": "انطلق في الجري السريع للأمام مباشرة من وضعية القرفصاء",
                "instruction_en": "Spring forward into a run directly from a crouched position.",
                "category": "locomotion",
                "variations_ar": ["انطلق راكضاً من القرفصاء", "اهجم بالجري مباشرة من وضع القرفصاء", "تسارع في الركض انطلاقاً من الانحناء"]
            }
        else:
            return {
                "instruction_ar": "اخفض جسدك واقرفص مع ثني الركبتين وثبات القدمين",
                "instruction_en": "Crouch down low with deeply bent knees and stable balance.",
                "category": "whole_body_gestures",
                "variations_ar": ["اتخذ وضعية القرفصاء", "انحنِ بقامتك لأسفل واقرفص", "اثنِ الركبتين في وقفة قرفصاء ثابتة"]
            }

    if 'hop' in base_lower or 'leap' in base_lower or 'skip' in base_lower:
        action_name = "الحجل" if "hop" in base_lower else ("الوثب" if "leap" in base_lower else "التخطي الخفيف")
        action_en = "hop" if "hop" in base_lower else ("leap" if "leap" in base_lower else "skip")
        return {
            "instruction_ar": f"امشِ للأمام ثم قم بحركة {action_name} ثم واصل المشي",
            "instruction_en": f"Walk forward, perform a dynamic {action_en}, and resume walking.",
            "category": "locomotion",
            "variations_ar": [f"امشِ واقفز حركة {action_name} ثم تابع السير", f"تخلل مشيك بحركة {action_name} رشيقة", f"تحرك بمشية متخللة بـ {action_name}"]
        }

    if 'stand' in base_lower:
        if 'walk_to_stand' in base_lower:
            return {
                "instruction_ar": "امشِ للأمام ثم تباطأ وتوقف في وضعية الوقوف الثابت",
                "instruction_en": "Walk forward, slow down, and come to a complete stable stand.",
                "category": "locomotion",
                "variations_ar": ["سر للأمام ثم قف تماماً", "امشِ وتوقف في وقفة معتدلة", "أنهِ مشيك بالوقوف الثابت"]
            }
        elif 'stand_to_walk' in base_lower:
            return {
                "instruction_ar": "ابدأ الحركة من وضعية الوقوف وانطلق في المشي للأمام",
                "instruction_en": "Start from a stationary standing posture and begin walking forward.",
                "category": "locomotion",
                "variations_ar": ["انطلق في المشي من وقفتك", "تحرك من وضع الثبات إلى السير للأمام", "ابدأ المشي قدماً"]
            }
        elif 'run_to_stand' in base_lower:
            return {
                "instruction_ar": "اركض للأمام ثم اخفض سرعتك وقف بثبات",
                "instruction_en": "Run forward, decelerate smoothly, and come to a halt standing up.",
                "category": "locomotion",
                "variations_ar": ["اجرِ ثم توقف واقفاً", "توقف عن الركض وقف باعتدال", "أوقف الجري واثبت في مكانك واقفاً"]
            }
        elif 'stand_to_run' in base_lower:
            return {
                "instruction_ar": "انطلق في الجري للأمام مباشرة من وضعية الوقوف",
                "instruction_en": "Accelerate into a run directly from standing still.",
                "category": "locomotion",
                "variations_ar": ["ابدأ الركض من وضع الوقوف", "انطلق بالجري سريعاً وأنت واقف", "تحول من الوقوف إلى الركض للأمام"]
            }
        else:
            return {
                "instruction_ar": "قف معتدلاً بوضعية طبيعية ومتوازنة",
                "instruction_en": "Stand upright with a natural, stable, and balanced posture.",
                "category": "whole_body_gestures",
                "variations_ar": ["قف منتصب القامة بثبات", "اثبت في وضعية الوقوف الطبيعي", "قف باعتدال واتزان"]
            }

    if 'sway' in base_lower:
        return {
            "instruction_ar": "تمايل بالجذع والجسد يمنة ويسرة برقة وهدوء وأنت واقف",
            "instruction_en": "Gently sway upper body side to side while standing in place.",
            "category": "whole_body_gestures",
            "variations_ar": ["تمايل ببطء يميناً وشمالاً", "تأرجح برفق في مكانك", "حرك جسدك بتمايل جانبي هادئ"]
        }

    if 'swing' in base_lower:
        return {
            "instruction_ar": "أرجح الذراعين للأمام والخلف بانسيابية أثناء الوقوف",
            "instruction_en": "Swing arms back and forth rhythmically while standing.",
            "category": "whole_body_gestures",
            "variations_ar": ["لوح بذراعيك ذهاباً وإياباً", "حرك الذراعين بأرجحة متناسقة", "أرجح يديك أثناء الوقوف"]
        }

    if 'look' in base_lower:
        return {
            "instruction_ar": "التفت بالرأس والجذع واستطلع محيطك في مختلف الاتجاهات",
            "instruction_en": "Look around and scan the environment by turning head and upper torso.",
            "category": "whole_body_gestures",
            "variations_ar": ["انظر حولك واستطلع المكان", "التفت يميناً ويساراً للمراقبة", "استكشف محيطك بالنظر حولك"]
        }

    if 'cartwheel' in base_lower:
        return {
            "instruction_ar": "أدِّ حركة العجلة البهلوانية الجانبية (كارت ويل) بركيزة اليدين ودوران الساقين",
            "instruction_en": "Perform a dynamic gymnastic cartwheel with hand support and leg rotation.",
            "category": "acrobatics",
            "variations_ar": ["نفذ عجلة جمباز جانبية بهلوانية", "أدِّ حركة كارت ويل رشيقة", "اقلب الجسد جانبياً في عجلة بهلوانية"]
        }

    if 'conversation_gestures' in base_lower or 'gestures' in base_lower:
        return {
            "instruction_ar": "تحرك بإيماءات تعبيرية طبيعية باليدين والجسد كأنك تخوض محادثة",
            "instruction_en": "Perform natural conversational hand and body gestures as if speaking.",
            "category": "whole_body_gestures",
            "variations_ar": ["لوح بيديك بإيماءات الحوار والتحدث", "عبر بحركات جسد وإيماءات كأنك تتكلم", "أظهر إيماءات تفاعلية للمحادثة"]
        }

    if 'urban' in base_lower or 'random_stuff' in base_lower or 'wait' in base_lower:
        return {
            "instruction_ar": "قف وانتظر بوضعية مسترخية مع حركات تبديل الوزن وإيماءات عفوية",
            "instruction_en": "Wait casually in place with subtle weight shifting and relaxed gestures.",
            "category": "whole_body_gestures",
            "variations_ar": ["قف بانتظار واسترخاء في مكانك", "بدل وزنك بين القدمين بتمهل", "قف بوقفة استراحة عفوية"]
        }

    if 'walk_to_run' in base_lower:
        return {
            "instruction_ar": "ابدأ بالمشي ثم زد سرعتك تدريجياً وتحول إلى الركض",
            "instruction_en": "Start walking and smoothly accelerate into a full run.",
            "category": "locomotion",
            "variations_ar": ["تدرج من المشي إلى الركض السريع", "سر ثم تسارع في الجري", "ابدأ مشياً وتسارع للركض"]
        }

    if 'run_to_walk' in base_lower:
        return {
            "instruction_ar": "كن راكضاً ثم هدئ سرعتك تدريجياً وتحول إلى المشي",
            "instruction_en": "Decelerate smoothly from running down to a steady walk.",
            "category": "locomotion",
            "variations_ar": ["ابطئ من الركض إلى المشي الهادئ", "انتقل من الجري إلى السير العادي", "هدئ سرعتك وتحول للمشي"]
        }

    if 'run' in base_lower:
        return {
            "instruction_ar": "اركض للأمام بخطى متسارعة ونشيطة",
            "instruction_en": "Run forward with active, energetic strides.",
            "category": "locomotion",
            "variations_ar": ["اجرِ للأمام بنشاط", "اركض مستقيماً للأمام", "واصل الجري السريع للأمام"]
        }

    return {
        "instruction_ar": "امشِ للأمام بخطوات ثابتة ومتزنة",
        "instruction_en": "Walk forward with steady, balanced strides.",
        "category": "locomotion",
        "variations_ar": ["سر للأمام بثبات", "امشِ قدماً بخطوات متزنة", "تقدم مشياً للأمام"]
    }

records = []
total_frames = 0

print("Downloading and processing 174 motions...")
for idx, rfilename in enumerate(pkl_files):
    local_path = hf_hub_download(repo_id=SOURCE_REPO, filename=rfilename, repo_type="dataset", local_dir=CACHE_DIR)
    try:
        with open(local_path, "rb") as f:
            data = pickle.load(f)
    except Exception:
        import joblib
        data = joblib.load(local_path)
        
    first_key = list(data.keys())[0]
    inner = data[first_key]
    
    dof = inner["dof"].astype(np.float32)                   # (T, 23)
    root_trans = inner["root_trans_offset"].astype(np.float32) # (T, 3)
    root_rot = inner["root_rot"].astype(np.float32)         # (T, 4)
    contact_mask = inner["contact_mask"].astype(np.float32) # (T, 2)
    fps = int(inner.get("fps", 30))
    num_frames = dof.shape[0]
    duration_sec = round(num_frames / float(fps), 3)
    total_frames += num_frames
    
    meta = generate_motion_metadata(rfilename)
    
    records.append({
        "episode_id": idx,
        "motion_file": rfilename,
        "motion_name": os.path.splitext(os.path.basename(rfilename))[0],
        "category": meta["category"],
        "instruction_ar": meta["instruction_ar"],
        "instruction_en": meta["instruction_en"],
        "variations_ar": meta["variations_ar"],
        "fps": fps,
        "num_frames": num_frames,
        "duration_seconds": duration_sec,
        "dof_trajectory": dof.tolist(),
        "root_trans_offset": root_trans.tolist(),
        "root_rot": root_rot.tolist(),
        "contact_mask": contact_mask.tolist(),
    })
    
    if (idx + 1) % 25 == 0 or (idx + 1) == len(pkl_files):
        print(f"Processed {idx + 1}/{len(pkl_files)} motions ({total_frames} frames so far)...")

df = pd.DataFrame(records)
print(f"Total motions processed: {len(df)}")
print(f"Total motion frames: {total_frames:,} ({total_frames / (30 * 3600):.2f} hours at 30 FPS)")
print("Category breakdown:\n", df['category'].value_counts())

# Stratified train/val split (90% train, 10% test)
np.random.seed(42)
test_indices = []
for cat, group in df.groupby('category'):
    n_test = max(1, int(len(group) * 0.10))
    test_indices.extend(group.sample(n=n_test, random_state=42).index.tolist())

train_df = df.drop(index=test_indices).reset_index(drop=True)
test_df = df.loc[test_indices].reset_index(drop=True)

print(f"Train split: {len(train_df)} motions")
print(f"Test split: {len(test_df)} motions")

OUT_DIR = "/tmp/ar_unitree_g1_pkg"
os.makedirs(f"{OUT_DIR}/data", exist_ok=True)

train_path = f"{OUT_DIR}/data/train.parquet"
test_path = f"{OUT_DIR}/data/test.parquet"

print("Saving Parquet files...")
train_df.to_parquet(train_path, index=False, compression="snappy")
test_df.to_parquet(test_path, index=False, compression="snappy")
print(f"Train parquet size: {os.path.getsize(train_path) / (1024*1024):.2f} MB")
print(f"Test parquet size: {os.path.getsize(test_path) / (1024*1024):.2f} MB")

# Create rich README.md dataset card
README_CONTENT = f"""---
license: mit
task_categories:
- robotics
tags:
- robotics
- humanoid
- unitree-g1
- whole-body-control
- motion-capture
- arabic
- fastvla
- flow-matching
language:
- ar
- en
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.parquet
  - split: test
    path: data/test.parquet
---

# 🤖 مجموعة بيانات حركات روبوت يونيتري G1 باللغة العربية
# Arabic Unitree G1 Humanoid Robot Motion Dataset (`ar-unitree-g1-motions`)

مجموعة بيانات متكاملة ومترجمة بدقة فائقة للفنون القتالية، والرقصات الشعبية والمعاصرة، والملاحة وحركات الجسد الكاملة لروبوت **Unitree G1** البشري (23 درجة حرية - 23 DoF) مقترنة بأوامر لغوية طبيعية باللغة العربية الفصحى بالإضافة إلى الإنجليزية.

A curated, production-ready humanoid whole-body motion dataset for the **Unitree G1 Humanoid Robot (23 DoF)** retargeted from high-fidelity mocap sources, paired with rich natural language instructions in **Modern Standard Arabic (الفصحى)** with multiple linguistic prompt variations, and English references.

---

## 📊 إحصائيات مجموعة البيانات (Dataset Statistics)

| المقياس (Metric) | القيمة (Value) |
|---|---|
| **إجمالي التسلسلات الحركية (Motions)** | **174 تسلسلاً حركياً كاملاً** |
| **إجمالي الإطارات (Total Frames)** | **{total_frames:,} إطار** |
| **المدة الزمنية الإجمالية (Duration)** | **{total_frames / (30 * 3600):.2f} ساعة ({total_frames / 30:.1f} ثانية)** |
| **معدل الإطارات (Sampling Rate)** | **30 هرتز (30 FPS)** |
| **درجات حرية التحكم (Action Space)** | **23 درجة حرية مشتركة (23-DoF Joint Angles)** |
| **حجم مجموعة التدريب (Train Split)** | **{len(train_df)} تسلسل (~90%)** |
| **حجم مجموعة الاختبار (Test Split)** | **{len(test_df)} تسلسل (~10%)** |
| **الأوامر اللغوية (Language Prompts)** | **عربي فصيح (`instruction_ar`) + تنويعات بلاغية (`variations_ar`) + إنجليزي (`instruction_en`)** |

---

## 🦾 تفصيل مفاصل روبوت Unitree G1 (23-DoF Actuation Table)

كل إطار في حقل `dof_trajectory` يمثل شعاعاً حركياً مستمراً أبعاده `(T, 23)` يغطي المفاصل التالية بدقة:

```
[0..5]   Left Leg:   Hip Pitch, Hip Roll, Hip Yaw, Knee, Ankle Pitch, Ankle Roll
[6..11]  Right Leg:  Hip Pitch, Hip Roll, Hip Yaw, Knee, Ankle Pitch, Ankle Roll
[12]     Waist:      Waist Yaw
[13..17] Left Arm:   Shoulder Pitch, Shoulder Roll, Shoulder Yaw, Elbow Pitch, Elbow Roll
[18..22] Right Arm:  Shoulder Pitch, Shoulder Roll, Shoulder Yaw, Elbow Pitch, Elbow Roll
```

بالإضافة إلى:
- `root_trans_offset`: إزاحة الجذع ثلاثية الأبعاد `(T, 3)` بالمتر.
- `root_rot`: رباعيات التوجيه الزاوي للجذع `(T, 4)` (Quaternion).
- `contact_mask`: مصفوفة التلامس الثنائي للقدمين مع الأرض `(T, 2)` (يسار / يمين).

---

## 🥋 تصنيفات الحركات (Motion Categories)

تغطي مجموعة البيانات 7 أنماط حركية ديناميكية:

1. **فنون قتالية (Martial Arts / Kung Fu - 12 تسلسلاً)**:
   - وضعية بروس لي القتالية الشهيرة.
   - اللكمات الخطافية المتتالية ودوران الجذع.
   - وضعية ركوب الخيل الثابتة (مابو Ma Bu) مع اللكمات المستقيمة.
   - الركلات الدائرية القوية (Roundhouse kicks) والركلات الجانبية (Side kicks).
2. **رقص وفولكلور تعبيري (Dance & Choreography - 16 تسلسلاً)**:
   - رقصات تراثية متوسطية ويونانية: هاسابوسيرفيكو (Hasaposerviko)، سيرتوس (Syrtos)، تاتسيا (Tatsia)، تساميكو (Tsamiko)، وزيبكيكو (Zeibekiko).
   - رقصات معاصرة: آر أند بي (R&B contemporary)، ورقصة تشارلستون الكلاسيكية.
   - رقصة الكابويرا البرازيلية القتالية مع التمايل المستمر (Ginga).
3. **حركات وملاحة المشي والجري (Locomotion - 83 تسلسلاً)**:
   - المشي للأمام، المشي للخلف، الالتفاف لليسار واليمين بزوايا 45° و 90° و 135° و 180°.
   - الجري السريع للأمام، الركض المكثف (Sprinting)، والخطوات الجانبية السريعة.
   - الزحف العادي والزحف العسكري الملاصق للأرض للأمام والخلف.
   - الوثب، الحجل، والقفز العمودي بكلتا القدمين مع الهبوط المتوازن.
4. **التعامل مع الصناديق والأجسام (Manipulation - 15 تسلسلاً)**:
   - الانحناء والتقاط الصناديق من الأرض بكلتا اليدين.
   - رفع الصناديق والمشي بها لمسافات متفاوتة ثم وضعها على الأرض بعناية.
5. **إيماءات وحركات الجسد بالكامل (Whole-Body Gestures - 41 تسلسلاً)**:
   - الوقوف، التمايل الهادئ، أرجحة الذراعين، واستطلاع المحيط بالالتفات.
   - السقوط المتحكم به والنهوض الذاتي (Fall and Get Up).
   - الانتقال بين الاستلقاء التام والقرفصاء والوقوف.
6. **حركات تعبيرية عاطفية (Expressive Motion - 6 تسلسلات)**:
   - حركات الجسد المعبرة عن الخوف، الغضب، الحماس، السعادة، الاسترخاء، والحزن.
7. **حركات بهلوانية (Acrobatics - تسلسل واحد)**:
   - حركة العجلة البهلوانية الجانبية (Cartwheel).

---

## 💻 كيفية الاستخدام في بايثون (Quickstart Usage)

### باستخدام مكتبة `datasets` من Hugging Face:
```python
from datasets import load_dataset
import numpy as np

# تحميل مجموعة البيانات بالكامل
dataset = load_dataset("{TARGET_REPO}")

train_split = dataset["train"]
print(f"عدد تسلسلات التدريب: {{len(train_split)}}")

# قراءة تسلسل حركي
sample = train_split[0]
print(f"الأمر بالعربية: {{sample['instruction_ar']}}")
print(f"التنويعات اللغوية: {{sample['variations_ar']}}")
print(f"التصنيف: {{sample['category']}}")

# تحويل مسار المفاصل إلى مصفوفة نمباي (T, 23)
dof_traj = np.array(sample["dof_trajectory"])
print(f"شكل مصفوفة المفاصل للـ Unitree G1: {{dof_traj.shape}} (23 مفصل)")
```

### الاستخدام في نماذج الرؤية واللغة والتحكم (FastVLA / LeRobot):
```python
import torch

# تغذية النموذج بالأمر العربي ومسار المفاصل للتحكم التنبؤي المتدفق (Flow Matching)
instruction = sample["instruction_ar"]
action_chunk = torch.tensor(sample["dof_trajectory"][:16], dtype=torch.float32) # نافذة تحكم 16 خطوة
print(f"Action chunk shape: {{action_chunk.shape}}") # [16, 23]
```

---

## 📜 الترخيص والمصادر (License & Attribution)
- المصدر الأساسي للحركات: تم إعادة استهداف وتكييف البيانات الحركية الأصلية من مستودع [`openhe/g1-retargeted-motions`](https://huggingface.co/datasets/openhe/g1-retargeted-motions) و ACCAD و LaFAN1 ومصادر موكاب معتمدة.
- الترخيص: **MIT License**.
- إعداد وتوثيق وترجمة وتطوير: **حمزة بوعجيلة (Hamza Bouajila)** لصالح إطار **FastVLA**.
"""

with open(f"{OUT_DIR}/README.md", "w", encoding="utf-8") as f:
    f.write(README_CONTENT.strip())

print("README.md dataset card written.")

# Upload to Hugging Face Hub
print(f"Uploading files to Hugging Face Hub repository {TARGET_REPO}...")
api.upload_folder(
    folder_path=OUT_DIR,
    repo_id=TARGET_REPO,
    repo_type="dataset",
    commit_message="Publish Arabic Unitree G1 Humanoid Motion Dataset (174 sequences, 23-DoF)",
)

print(f"🎉 SUCCESS! Dataset published at: https://huggingface.co/datasets/{TARGET_REPO}")
