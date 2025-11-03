<p align="center">
  <img src="https://github.com/user-attachments/assets/64d9919e-ff96-4588-ae68-75c3f22b160e" 
       alt="Strivio logo" width="400" height="400" />
</p>

# **BitePulse AI** 

This project is the Capstone Project (AAI-590) for [the M.S. in Applied Artificial Intelligence](https://onlinedegrees.sandiego.edu/masters-applied-artificial-intelligence/) at [the University of San Diego (USD)](https://www.sandiego.edu/). 

-- **Project Status: Ongoing**

## **Introduction**

BitePulse AI solves a problem most people never notice. We eat faster than we think. Fast eating, meaning repeated bites or sips with almost no pause, is linked to bloating, reflux, weaker fullness signals, and overeating, because the body does not get enough time to register "stop".
Researchers already treat eating pace as real data. They track "intake events" (each bite or sip) on video and measure eating pace, because pace is connected to self-control and comfort.
In daily life, there is no feedback loop. Watches track steps and heart rate, food apps track calories but nothing tells us, “You just took 12 bites in 40 seconds”.
BitePulse AI changes that. It looks at a short meal video, analyzes how a person is eating in real time, and turns that into simple feedback they can act on. The goal is not dieting or shame. The goal is awareness, slow down a bit, feel better after meals, and stay in control of eating instead of running on autopilot.

## **Objective**

The goal of BitePulse AI is simple, an AI eating pace coach that watches a short meal clip and gives instant, human feedback. The system detects intake events (bites, sips), measures how fast the user is eating, finds burst eating (back-to-back bites with almost no pause), and notices when they slow down and chew. It then turns that into plain language, for example, "you ate in nonstop bursts for 28 seconds, try adding small pauses so your body can register fullness”.
The finished demo is a Streamlit app (or an iOS app as a farther stretch goal). The user loads a meal clip, sees the video, a timeline of bites, pace metrics, and friendly coaching. That demo can become a consumer app (meal pace feedback in under 1 minute), or an API for wellness platforms that want behavior data without building their own computer vision.

## **BitePulse AI Usefulness**

This is important for three reasons:
* Comfort: Eating too fast can leave someone bloated, uncomfortable, and still eating before the body has time to say stop. Slowing down between bites is already recommended in nutrition and mindful eating programs.
* We change what we can see: We track sleep, we track steps, but no one tracks eating pace. BitePulse AI gives an eating pace score, so users can actually adjust in real life.
* It is not about shame. BitePulse AI does not count calories or judge bodies. It gives neutral feedback, for example, “you ate in a 30 second nonstop burst, try pausing so your body can feel full”. The goal is comfort and control, not diet culture.

## **BitePulse AI Audience**
BitePulse AI serves both consumers (B2C) and wellness partners (B2B):
* For consumers (B2C), it helps everyday eaters who want better comfort after meals, calmer digestion, and more mindful control. A user can record a short meal clip and get clear feedback on their eating pace.
* For wellness partners (B2B), like nutrition coaches and digital health platforms, BitePulse AI delivers objective eating behavior signals (for example, slow pauses versus nonstop bursts) through an API, without anyone having to watch the video. In both cases, the goal is awareness and coaching, not medical diagnosis.

## **Data Source and Privacy**
BitePulse AI is trained on [EatSense](https://groups.inf.ed.ac.uk/vision/DATASETS/EATSENSE/), a research dataset of real people eating real meals, with every frame labeled for actions like eating, chewing, drinking, or resting. Faces are anonymized. This lets us train deep learning models to detect bites and pacing patterns without recording new users.
In the real product, BitePulse AI can run privately on device for consumers, or through a secure backend for wellness partners. For this capstone, we only use the public EatSense dataset, and we do not collect or store new personal video.

## **Methods Used**

* 
* 


## **Technologies**
*
*


## **License**

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.


## **Acknowledgments**

Thanks to Professor Anna Marbut for guidance and feedback.
