# YouTube Reactions to Elon Musk: A Sentiment Analysis Case Study

## Overview
This project began as an exploration of YouTube reactions to Elon Musk's controversial gesture, widely interpreted as a Nazi salute. However, it quickly transformed into an investigation of the limitations of NLP-based sentiment analysis tools—VADER, TextBlob, and BERT—respecting accurately capturing the meaning and context of YouTube comments.

### Channels Mapped:
- **DW**: [Watch here](https://www.youtube.com/watch?v=48gTx8MpRmI&t=40s&ab_channel=DWNews)
- **Guardian News**: [Watch here](https://www.youtube.com/watch?v=smQNNo2a9xc&ab_channel=GuardianNews)
- **MSNBC**: [Watch here](https://www.youtube.com/watch?v=_J-uvMGfpTA&ab_channel=MSNBC)
- **Ben Shapiro**: [Watch here](https://www.youtube.com/watch?v=RZy5fRlT14c&t=67s&ab_channel=BenShapiro)
- **Candace Owens**: [Watch here](https://www.youtube.com/watch?v=aN684idlXxA&t=30s&ab_channel=CandaceOwens)
- **Rubin Report**: [Watch here](https://www.youtube.com/watch?v=72-qe4OjCj0&t=91s&ab_channel=TheRubinReport)

## Motivation
While sentiment analysis tools are widely used to gauge public opinion, they often fail to capture nuances such as sarcasm, political subtext, and cultural references. This project examines how these tools perform on real-world, high-stakes commentary and where they fall short.

## Tools Used
- **VADER** (Valence Aware Dictionary and Sentiment Reasoner): A lexicon-based approach designed for social media analysis.
- **TextBlob**: A simple NLP tool for basic sentiment analysis.
- **BERT** (Bidirectional Encoder Representations from Transformers): A deep-learning-based NLP model capable of understanding context.

## Findings
1. **Contextual Failures**: None of the models effectively captured the full meaning behind sarcastic, coded, or politically charged comments.
2. **Polarization Gaps**: Extreme opinions were often misclassified as neutral or mild sentiment.
3. **Cultural Blind Spots**: References to historical events and ideological symbols were often misunderstood or ignored by the models.

## Methodology
1. **Data Collection**: API extracted comments from YouTube videos discussing Elon Musk’s gesture.
2. **Sentiment Analysis**: Applied VADER, TextBlob, and BERT (RoBERTa) to classify comments.
3. **Human Evaluation**: Compared model results with manual annotations to assess accuracy.

## Challenges
- **Meme Culture & Slang**: Many comments relied on internet-specific language that NLP models failed to interpret correctly.
- **Subtlety of Sarcasm**: Most tools struggled with sarcastic or ironic remarks, misclassifying them as positive or neutral.
- **Political Language Complexity**: NLP tools lack awareness of evolving political discourse and coded language.

## Conclusion
While NLP tools like VADER, TextBlob, and BERT offer valuable insights in standard sentiment analysis tasks, they struggle with highly nuanced, politically charged, and context-dependent discussions. This project highlights the need for more advanced models capable of deeper contextual understanding and cultural awareness.

## Future Work
- Fine-tune BERT with a dataset specifically labeled for sarcasm, political discourse, and coded language.
- Explore multimodal approaches that incorporate video and audio cues for better sentiment detection.
- Develop a hybrid model combining human moderation with AI to improve contextual understanding.

## How to Use This Repository
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/youtube-elon-nlp.git
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run sentiment analysis:
    ```bash
    python analyze_sentiment.py
    ```

## Contributions
Contributions are welcome! If you have ideas for improving sentiment analysis in complex contexts, feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License.
