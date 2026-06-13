"""
Dataset builder for NLP Wellness Assistant.
Creates a labeled dataset combining Bhagavad Gita verses + synthetic Puranic teachings,
mapped to emotion categories for training.
"""

import pandas as pd
import json
import os

# ─────────────────────────────────────────────
# Bhagavad Gita Verses (real translations)
# ─────────────────────────────────────────────
GITA_VERSES = [
    # STRESS
    {
        "source": "Bhagavad Gita",
        "chapter": 2, "verse": 14,
        "sanskrit": "mātrā-sparśās tu kaunteya śītoṣṇa-sukha-duḥkha-dāḥ",
        "english": "O son of Kunti, the contact between the senses and the sense objects gives rise to feelings of cold and heat, pleasure and pain. They are transient and fleeting; therefore, you must endure them.",
        "meaning": "Pain and pleasure are temporary sensations caused by contact with the material world. A wise person learns to endure them without being disturbed.",
        "emotion": "stress",
        "keywords": "pain pleasure temporary endure senses contact transient fleeting disturbed"
    },
    {
        "source": "Bhagavad Gita",
        "chapter": 2, "verse": 47,
        "sanskrit": "karmaṇy evādhikāras te mā phaleṣu kadācana",
        "english": "You have a right to perform your prescribed duties, but you are not entitled to the fruits of your actions. Never consider yourself the cause of the results of your activities, and never be attached to not doing your duty.",
        "meaning": "Focus only on your actions, not on the results. Detachment from outcomes reduces stress and anxiety.",
        "emotion": "stress",
        "keywords": "duty action results detachment focus work perform prescribed fruits attached"
    },
    {
        "source": "Bhagavad Gita",
        "chapter": 6, "verse": 17,
        "sanskrit": "yuktāhāra-vihārasya yukta-ceṣṭasya karmasu",
        "english": "He who is regulated in his habits of eating, sleeping, recreation and work can mitigate all material pains by practicing the yoga system.",
        "meaning": "A balanced lifestyle — regulated eating, sleep, and work — is the foundation of mental peace and physical wellness.",
        "emotion": "stress",
        "keywords": "regulated eating sleeping recreation work balance lifestyle mental peace wellness mitigate"
    },
    {
        "source": "Bhagavad Gita",
        "chapter": 18, "verse": 66,
        "sanskrit": "sarva-dharmān parityajya mām ekaṁ śaraṇaṁ vraja",
        "english": "Abandon all varieties of religion and just surrender unto Me. I shall deliver you from all sinful reactions. Do not fear.",
        "meaning": "Let go of all burdens and surrender to the divine. Fear and stress dissolve when you place your trust in a higher power.",
        "emotion": "stress",
        "keywords": "surrender fear burden trust divine deliver abandon sinful reactions peace"
    },
    {
        "source": "Bhagavad Gita",
        "chapter": 5, "verse": 10,
        "sanskrit": "brahmaṇy ādhāya karmāṇi saṅgaṁ tyaktvā karoti yaḥ",
        "english": "One who performs his duty without attachment, surrendering the results unto the Supreme Lord, is unaffected by sinful action, as the lotus leaf is untouched by water.",
        "meaning": "Like a lotus leaf untouched by water, perform your duties without attachment and remain unaffected by stress and outcomes.",
        "emotion": "stress",
        "keywords": "duty attachment unaffected results lotus untouched perform sinful action"
    },

    # ANXIETY
    {
        "source": "Bhagavad Gita",
        "chapter": 2, "verse": 56,
        "sanskrit": "duḥkheṣv anudvigna-manāḥ sukheṣu vigata-spṛhaḥ",
        "english": "One who is not disturbed in mind even amidst the threefold miseries or elated when there is happiness, and who is free from attachment, fear and anger, is called a sage of steady mind.",
        "meaning": "True wisdom lies in maintaining equanimity — neither anxious in suffering nor overjoyed in happiness. Steady mind is the goal.",
        "emotion": "anxiety",
        "keywords": "anxiety fear anger miseries happiness steady mind equanimity disturbed elated attachment"
    },
    {
        "source": "Bhagavad Gita",
        "chapter": 4, "verse": 40,
        "sanskrit": "ajñaś cāśraddadhānaś ca saṁśayātmā vinaśyati",
        "english": "But ignorant and faithless persons who doubt the revealed scriptures do not attain God consciousness; they fall down. For the doubting soul there is happiness neither in this world nor in the next.",
        "meaning": "Doubt destroys peace. A person consumed by anxiety and doubt finds no happiness. Faith in oneself and one's path removes anxiety.",
        "emotion": "anxiety",
        "keywords": "doubt faith anxiety ignorant faithless consciousness happiness destroyed worry uncertain"
    },
    {
        "source": "Bhagavad Gita",
        "chapter": 6, "verse": 35,
        "sanskrit": "asaṁśayaṁ mahā-bāho mano durnigrahaṁ calam",
        "english": "Lord Krishna said: O mighty-armed son of Kunti, it is undoubtedly very difficult to curb the restless mind, but it is possible by suitable practice and by detachment.",
        "meaning": "The restless, anxious mind can be controlled through consistent practice and detachment from outcomes. Anxiety can be overcome.",
        "emotion": "anxiety",
        "keywords": "restless mind control practice detachment anxiety difficult overcome curb steady"
    },
    {
        "source": "Bhagavad Gita",
        "chapter": 2, "verse": 66,
        "sanskrit": "nāsti buddhir ayuktasya na cāyuktasya bhāvanā",
        "english": "One who is not connected with the Supreme can have neither transcendental intelligence nor a steady mind, without which there is no possibility of peace. And how can there be any happiness without peace?",
        "meaning": "Without inner peace, there is no happiness. Anxiety arises from an unsteady mind. Reconnecting with your inner self restores peace.",
        "emotion": "anxiety",
        "keywords": "peace happiness steady mind intelligence anxiety restless unsteady reconnect inner"
    },
    {
        "source": "Vishnu Purana",
        "chapter": 1, "verse": 17,
        "sanskrit": "bhayaṁ dvitīyābhiniveśataḥ syāt",
        "english": "Fear arises when a living entity misidentifies himself with the material body because of absorption in the external illusory energy.",
        "meaning": "Anxiety and fear arise from over-identification with the temporary body and material world. Your true self is eternal and fearless.",
        "emotion": "anxiety",
        "keywords": "fear anxiety body material temporary eternal fearless identify illusion external"
    },

    # MOTIVATION
    {
        "source": "Bhagavad Gita",
        "chapter": 3, "verse": 21,
        "sanskrit": "yad yad ācarati śreṣṭhas tat tad evetaro janaḥ",
        "english": "Whatever action a great man performs, common men follow. And whatever standards he sets by exemplary acts, all the world pursues.",
        "meaning": "Be the person who leads by example. Your actions inspire those around you. Rise up and act with purpose and greatness.",
        "emotion": "motivation",
        "keywords": "action great example inspire lead purpose follow standards world pursues"
    },
    {
        "source": "Bhagavad Gita",
        "chapter": 18, "verse": 48,
        "sanskrit": "saha-jaṁ karma kaunteya sa-doṣam api na tyajet",
        "english": "Every endeavor is covered by some fault, just as fire is covered by smoke. Therefore one should not give up the work born of his nature, O son of Kunti, even if such work is full of fault.",
        "meaning": "Don't give up because of imperfection. Every effort has flaws, but consistent action toward your purpose is what matters.",
        "emotion": "motivation",
        "keywords": "effort action purpose persist consistency imperfection give up nature work endeavor"
    },
    {
        "source": "Bhagavad Gita",
        "chapter": 2, "verse": 3,
        "sanskrit": "klaibyaṁ mā sma gamaḥ pārtha naitat tvayy upapadyate",
        "english": "O Partha, do not yield to this degrading impotence. It does not become you. Give up such petty weakness of heart and arise, O vanquisher of enemies.",
        "meaning": "Arise! Do not succumb to weakness or lack of motivation. You have the strength within you. Stand up and fight for your purpose.",
        "emotion": "motivation",
        "keywords": "arise strength weakness impotence motivation rise fight purpose vanquish petty"
    },
    {
        "source": "Bhagavad Gita",
        "chapter": 11, "verse": 33,
        "sanskrit": "tasmāt tvam uttiṣṭha yaśo labhasva",
        "english": "Therefore get up. Prepare to fight and win glory. Conquer your enemies and enjoy a flourishing kingdom. They are already put to death by My arrangement, and you can be but an instrument in the fight.",
        "meaning": "Rise and act! Success awaits those who take action. You are capable of greatness — be the instrument of your own destiny.",
        "emotion": "motivation",
        "keywords": "rise act success greatness fight glory conquer win instrument destiny flourishing"
    },
    {
        "source": "Skanda Purana",
        "chapter": 3, "verse": 12,
        "sanskrit": "udyamaḥ sāhasaṁ dhairyaṁ buddhiḥ śaktiḥ parākramaḥ",
        "english": "Effort, courage, patience, intelligence, strength, and bravery — these six qualities are the characteristics of the one who achieves success.",
        "meaning": "The six pillars of achievement are: effort, courage, patience, intelligence, strength, and bravery. Cultivate them daily.",
        "emotion": "motivation",
        "keywords": "effort courage patience intelligence strength bravery success achieve qualities characteristics"
    },

    # CONFUSION
    {
        "source": "Bhagavad Gita",
        "chapter": 4, "verse": 34,
        "sanskrit": "tad viddhi praṇipātena paripraśnena sevayā",
        "english": "Just try to learn the truth by approaching a spiritual master. Inquire from him submissively and render service unto him. The self-realized souls can impart knowledge unto you because they have seen the truth.",
        "meaning": "When confused, seek guidance from a wise teacher. Approach with humility, ask questions, and serve. Knowledge dispels confusion.",
        "emotion": "confusion",
        "keywords": "confused guidance teacher knowledge learn truth inquire wisdom self-realized humble seek"
    },
    {
        "source": "Bhagavad Gita",
        "chapter": 10, "verse": 10,
        "sanskrit": "teṣāṁ satata-yuktānāṁ bhajatāṁ prīti-pūrvakam",
        "english": "To those who are constantly devoted to serving Me with love, I give the understanding by which they can come to Me.",
        "meaning": "To those who seek sincerely with devotion, clarity and understanding naturally come. Trust the process of sincere seeking.",
        "emotion": "confusion",
        "keywords": "clarity understanding devoted sincere seek clarity knowledge confused uncertain devotion"
    },
    {
        "source": "Bhagavad Gita",
        "chapter": 3, "verse": 2,
        "sanskrit": "vyāmiśreṇeva vākyena buddhiṁ mohayasīva me",
        "english": "My intelligence is bewildered by Your equivocal instructions. Therefore, please tell me decisively which will be most beneficial for me.",
        "meaning": "Even the greatest warriors felt confused. It is okay to feel lost. Ask clearly for what you need to know, and clarity will come.",
        "emotion": "confusion",
        "keywords": "confused bewildered clarity lost uncertain ask direction beneficial intelligence instruction"
    },
    {
        "source": "Bhagavad Gita",
        "chapter": 18, "verse": 61,
        "sanskrit": "īśvaraḥ sarva-bhūtānāṁ hṛd-deśe arjuna tiṣṭhati",
        "english": "The Supreme Lord is situated in everyone's heart, O Arjuna, and is directing the wanderings of all living entities, who are seated as on a machine, made of the material energy.",
        "meaning": "When confused about your path, look within. The answer often lies in your heart, not outside in the world's noise.",
        "emotion": "confusion",
        "keywords": "confused path direction heart within inner voice lost uncertain wandering divine"
    },
    {
        "source": "Markandeya Purana",
        "chapter": 5, "verse": 8,
        "sanskrit": "viveko jñāna-sambhavaḥ",
        "english": "Discernment is born of knowledge. When we cultivate knowledge with patience, the fog of confusion lifts and the right path becomes visible.",
        "meaning": "Confusion lifts when you invest in learning and developing discernment. Study, reflect, and the right path will reveal itself.",
        "emotion": "confusion",
        "keywords": "confusion knowledge discernment learning reflect path clear fog patient study cultivate"
    },

    # HAPPINESS
    {
        "source": "Bhagavad Gita",
        "chapter": 5, "verse": 21,
        "sanskrit": "bāhya-sparśeṣv asaktātmā vindaty ātmani yat sukham",
        "english": "Such a liberated person is not attracted to material sense pleasure but is always in trance, enjoying the pleasure within. In this way the self-realized person enjoys unlimited happiness.",
        "meaning": "True happiness is not found in external pleasures but in the joy of inner peace and self-realization.",
        "emotion": "happiness",
        "keywords": "happiness joy inner peace pleasure external internal liberation fulfilled content bliss"
    },
    {
        "source": "Bhagavad Gita",
        "chapter": 6, "verse": 27,
        "sanskrit": "praśānta-manasaṁ hy enaṁ yoginaṁ sukham uttamam",
        "english": "The yogi whose mind is fixed on Me verily attains the highest perfection of transcendental happiness. He is beyond the mode of passion, he realizes his qualitative identity with the Supreme, and thus he is freed from all reactions to past deeds.",
        "meaning": "The highest happiness comes from a peaceful, focused mind. When the mind is calm, happiness flows naturally.",
        "emotion": "happiness",
        "keywords": "happiness peace joy mind calm perfection transcendental bliss content serene fulfilled"
    },
    {
        "source": "Bhagavad Gita",
        "chapter": 18, "verse": 37,
        "sanskrit": "yat tad agre viṣam iva pariṇāme amṛtopamam",
        "english": "That which in the beginning may be just like poison but at the end is just like nectar and which awakens one to self-realization is said to be happiness in the mode of goodness.",
        "meaning": "Real happiness often comes after difficult work and patience — like nectar after bitterness. Embrace the journey.",
        "emotion": "happiness",
        "keywords": "happiness joy patience nectar journey reward goodness awakening self-realization"
    },
    {
        "source": "Bhagavad Purana",
        "chapter": 11, "verse": 14,
        "sanskrit": "ānandaṁ brahmaṇo vidvān na bibheti kutaścana",
        "english": "One who knows the bliss of Brahman fears nothing and grieves for nothing. Such a person lives in a state of constant inner joy.",
        "meaning": "Inner joy is the natural state of a person connected to their true self. Cultivate that connection to sustain lasting happiness.",
        "emotion": "happiness",
        "keywords": "joy bliss happiness fearless grief inner state connected true self contentment divine"
    },

    # GENERAL WELLNESS
    {
        "source": "Bhagavad Gita",
        "chapter": 6, "verse": 5,
        "sanskrit": "uddhared ātmanātmānaṁ nātmānam avasādayet",
        "english": "One must deliver himself with the help of his mind, and not degrade himself. The mind is the friend of the conditioned soul, and his enemy as well.",
        "meaning": "Your mind can be your greatest ally or your worst enemy. Elevate yourself through discipline, self-awareness, and positive thinking.",
        "emotion": "general",
        "keywords": "mind self discipline wellbeing friend enemy elevate degrade positive awareness deliver"
    },
    {
        "source": "Bhagavad Gita",
        "chapter": 13, "verse": 8,
        "sanskrit": "amānitvam adambhitvam ahiṁsā kṣāntir ārjavam",
        "english": "Humility; pridelessness; nonviolence; tolerance; simplicity; approaching a bona fide spiritual master; cleanliness; steadiness; self-control — all these I declare to be knowledge.",
        "meaning": "True wellness encompasses humility, nonviolence, tolerance, cleanliness, and self-control. These are the foundations of a healthy life.",
        "emotion": "general",
        "keywords": "wellness health humility nonviolence tolerance cleanliness steadiness self-control simplicity knowledge"
    },
    {
        "source": "Bhagavad Gita",
        "chapter": 17, "verse": 8,
        "sanskrit": "āyuḥ-sattva-balārogya-sukha-prīti-vivardhanāḥ",
        "english": "Foods in the mode of goodness increase the duration of life, purify one's existence and give strength, health, happiness and satisfaction.",
        "meaning": "What you eat influences your mind and body. Foods in goodness — fresh, nutritious, and pure — promote health and mental clarity.",
        "emotion": "general",
        "keywords": "health food nutrition lifestyle body mind strength happiness satisfaction purify balance"
    },
    {
        "source": "Bhagavad Gita",
        "chapter": 4, "verse": 38,
        "sanskrit": "na hi jñānena sadṛśaṁ pavitram iha vidyate",
        "english": "In this world, there is nothing so sublime and pure as transcendental knowledge. Such knowledge is the mature fruit of all mysticism. And one who has become accomplished in the practice of devotional service enjoys this knowledge within himself in due course of time.",
        "meaning": "Self-knowledge is the purest treasure. Invest in learning about yourself and the world — this is the highest form of wellness.",
        "emotion": "general",
        "keywords": "knowledge self wisdom learning growth spiritual wellness sublime pure transcendental practice"
    },
    {
        "source": "Linga Purana",
        "chapter": 2, "verse": 5,
        "sanskrit": "dhyānaṁ sarva-śarīrasya auṣadham",
        "english": "Meditation is the medicine for the entire body and mind. One who meditates daily finds relief from the ailments of body, mind and spirit.",
        "meaning": "Daily meditation is one of the most powerful wellness practices. It heals the body, calms the mind, and nurtures the spirit.",
        "emotion": "general",
        "keywords": "meditation wellness health body mind spirit daily practice healing calm peace routine"
    },
    {
        "source": "Bhagavad Gita",
        "chapter": 2, "verse": 48,
        "sanskrit": "yoga-sthaḥ kuru karmāṇi saṅgaṁ tyaktvā dhanañjaya",
        "english": "Perform your duty equipoised, O Arjuna, abandoning all attachment to success or failure. Such equanimity is called yoga.",
        "meaning": "Do your work with balance and equanimity, free from attachment to results. This is the essence of yoga and true wellness.",
        "emotion": "general",
        "keywords": "balance equanimity yoga work duty attachment results success failure wellness"
    },
    {
        "source": "Bhagavad Purana",
        "chapter": 3, "verse": 25,
        "sanskrit": "sattvaṁ rajas tama iti guṇāḥ prakṛti-sambhavāḥ",
        "english": "Material nature consists of three modes: goodness, passion, and ignorance. When the living entity comes in contact with nature, he is conditioned by these modes.",
        "meaning": "Understand the three modes of nature within you. Cultivate goodness (sattva) through pure food, thoughts, and actions for optimal wellness.",
        "emotion": "general",
        "keywords": "nature modes goodness passion ignorance cultivate pure wellness balance mind body"
    },
]

# ─────────────────────────────────────────────
# Training phrases per emotion (for model training)
# ─────────────────────────────────────────────
TRAINING_PHRASES = {
    "stress": [
        "I am feeling very stressed out", "work pressure is overwhelming me",
        "I can't handle the stress anymore", "everything feels too much",
        "I am burned out", "too much pressure at work", "stress is killing me",
        "I feel overwhelmed by responsibilities", "deadlines are stressing me out",
        "I feel like I am breaking down", "my workload is unbearable",
        "I am exhausted and stressed", "constant stress is affecting my health",
        "I cannot relax no matter what", "stress has taken over my life",
        "I feel so much pressure", "too many things to handle at once",
        "I am under a lot of stress", "feeling mentally drained",
        "stress and tension everywhere", "life feels like too much to bear",
        "I need relief from stress", "the pressure never stops",
        "I feel tense all the time", "stress is making me sick",
    ],
    "anxiety": [
        "I am feeling anxious", "I have too much anxiety", "I feel scared about the future",
        "anxiety is taking over", "I am worried all the time", "fear and anxiety haunt me",
        "I feel nervous and restless", "I cannot stop worrying",
        "constant worry is draining me", "I feel a sense of dread",
        "I am anxious about my future", "panic attacks are frequent",
        "I feel fearful without reason", "anxiety is overwhelming me",
        "I am afraid of what comes next", "my heart races with anxiety",
        "I can't sleep because of worry", "anxiety about everything",
        "I feel uneasy and restless", "nervous and anxious all day",
        "I dread going to work", "anxiety is ruining my life",
        "I feel like something bad will happen", "I am scared and nervous",
        "overthinking and anxiety", "fear of failure is crushing me",
    ],
    "motivation": [
        "I have lost my motivation", "I feel like giving up",
        "I don't feel like doing anything", "I lack the drive to succeed",
        "I need motivation to continue", "feeling demotivated today",
        "I can't find the energy to work", "I want to achieve but don't know how",
        "I feel stuck and uninspired", "no motivation left in me",
        "I need inspiration to keep going", "how do I find my purpose",
        "I feel like a failure", "lost my passion for work",
        "I need a push to move forward", "struggling to stay motivated",
        "I have no will to do anything", "motivation seems impossible",
        "I want to succeed but can't start", "feeling useless and unproductive",
        "I want to do great things but feel lazy", "need encouragement",
        "I am procrastinating too much", "I feel like nothing matters",
        "how do I get back on track", "lost my drive and purpose",
    ],
    "confusion": [
        "I am confused about my future", "I don't know what to do with my life",
        "I feel lost and directionless", "confused about which path to take",
        "I can't decide what is right", "I am unable to make decisions",
        "everything feels uncertain", "I don't understand my purpose",
        "I need clarity about my career", "confused and unsure about everything",
        "I feel completely lost", "which direction should I go",
        "I don't know what is right for me", "confused about my relationships",
        "I need guidance on my path", "I feel like I am at a crossroads",
        "my mind is in chaos", "I can't figure out what I want",
        "I don't know what my next step should be", "feeling directionless",
        "I have too many choices and I'm confused", "my life has no clear direction",
        "I need someone to show me the way", "I feel mentally foggy",
        "I am lost in my thoughts", "confused and overwhelmed with decisions",
    ],
    "happiness": [
        "I am feeling happy today", "I feel joyful and content",
        "life feels wonderful right now", "I am grateful and happy",
        "feeling positive and energized", "I am in a great mood",
        "happiness fills my heart", "I feel blessed and joyful",
        "today is a beautiful day", "I feel at peace and happy",
        "I am excited about my life", "I feel full of joy",
        "everything is going well", "I am living my best life",
        "I feel uplifted and happy", "my heart is full of gratitude",
        "I am experiencing pure joy", "I feel wonderful today",
        "positive vibes only today", "I am happy and at peace",
        "life is beautiful and fulfilling", "I am cheerful and motivated",
        "I feel great today", "joy and happiness surround me",
        "I am thriving and flourishing", "bliss and contentment fill me",
    ],
    "general": [
        "how should I live well", "give me advice for better living",
        "I want to improve my lifestyle", "what is the meaning of life",
        "how can I be a better person", "I want to grow spiritually",
        "I seek wisdom for daily life", "how to maintain good health",
        "what does the Gita say about life", "I want to understand myself better",
        "how to develop good habits", "give me spiritual guidance",
        "I want to find inner peace", "how to improve my mental wellbeing",
        "I need wellness advice", "how to balance work and life",
        "I want to be more mindful", "how to cultivate positive energy",
        "what are the principles of a good life", "I want general life guidance",
        "how to be happier every day", "I seek knowledge and wisdom",
        "how to build a peaceful life", "teach me something meaningful",
        "I want to improve my overall wellbeing", "guide me toward a better life",
    ],
}


def create_dataset():
    rows = []

    # Add verse-keyword records
    for v in GITA_VERSES:
        rows.append({
            "text": v["keywords"],
            "emotion": v["emotion"],
            "source": v["source"],
            "chapter": v["chapter"],
            "verse": v["verse"],
            "sanskrit": v["sanskrit"],
            "english": v["english"],
            "meaning": v["meaning"],
            "type": "verse"
        })

    # Add training phrases
    for emotion, phrases in TRAINING_PHRASES.items():
        for phrase in phrases:
            rows.append({
                "text": phrase,
                "emotion": emotion,
                "source": "",
                "chapter": None,
                "verse": None,
                "sanskrit": "",
                "english": "",
                "meaning": "",
                "type": "training"
            })

    df = pd.DataFrame(rows)
    os.makedirs("dataset", exist_ok=True)
    df.to_csv("dataset/wellness_dataset.csv", index=False)

    # Separate verse database
    verse_df = pd.DataFrame([
        {
            "source": v["source"],
            "chapter": v["chapter"],
            "verse": v["verse"],
            "sanskrit": v["sanskrit"],
            "english": v["english"],
            "meaning": v["meaning"],
            "emotion": v["emotion"],
            "keywords": v["keywords"],
        }
        for v in GITA_VERSES
    ])
    verse_df.to_csv("dataset/verses_db.csv", index=False)

    print(f"✅ Dataset created: {len(df)} total records")
    print(f"✅ Verse DB: {len(verse_df)} verses")
    print(f"\nEmotion distribution:")
    print(df[df['type'] == 'training']['emotion'].value_counts())
    return df, verse_df


if __name__ == "__main__":
    create_dataset()
