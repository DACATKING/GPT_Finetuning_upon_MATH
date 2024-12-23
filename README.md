# GPT_Finetuning_upon_MATH

## Description

This project focuses on fine-tuning GPT-3.5 and GPT-4 models to enhance their mathematical problem-solving capabilities. By leveraging specialized datasets and training scripts, the aim is to improve the models' proficiency in understanding and solving mathematical problems.

## Repository Structure

- `data_process/`: Contains scripts for data preprocessing, including:
  - `data_cleaning.py`: Cleans and formats raw mathematical datasets.
  - `data_tokenization.py`: Tokenizes the cleaned data for model training.
- `fine_tune_gpt3.5_math_main.py`: Main script for fine-tuning the GPT-3.5 model on mathematical datasets.
- `fine_tune_gpt4omini_math_main.py`: Main script for fine-tuning the GPT-4 model on mathematical datasets.
- `README.md`: Provides an overview of the project and instructions for usage.

## Usage

To fine-tune the models, follow these steps:

1. **Data Preparation**: Preprocess the raw data using the scripts in the `data_process/` directory.
   ```bash
   python data_process/data_cleaning.py
   python data_process/data_tokenization.py
   ```
2. **Fine-Tuning**: Execute the appropriate fine-tuning script.
   - For GPT-3.5:
     ```bash
     python fine_tune_gpt3.5_math_main.py --data_path=processed_data.json --output_dir=gpt3.5_finetuned
     ```
   - For GPT-4:
     ```bash
     python fine_tune_gpt4omini_math_main.py --data_path=processed_data.json --output_dir=gpt4_finetuned
     ```
3. **Evaluation**: Assess the performance of the fine-tuned models using evaluation scripts (to be implemented).

## Results and Observations

### Performance Summary

1. **Improved Trigonometric Reasoning**
   - Fine-tuning GPT-3.5 Turbo using the MATH dataset significantly enhanced the model's ability to solve trigonometric problems. For instance:
     - **Trigonometric Problem Example**: In a triangle where angles form an arithmetic sequence, the fine-tuned model correctly calculated \( \sin \left( \frac{C - A}{2} \right) \), whereas the base GPT-3.5 Turbo provided incomplete or incorrect answers.

2. **General Problem Accuracy**
   - Overall, the fine-tuned GPT-3.5 Turbo exhibited mixed results:
     - **Base GPT-3.5 Turbo**: 58% accuracy.
     - **Fine-tuned GPT-3.5 Turbo**: 43% accuracy.
   - This decrease in overall accuracy suggests a trade-off, where improvements in specific problem types (like trigonometry) might come at the expense of general performance.

3. **Performance on HARDMath Dataset**
   - The HARDMath dataset evaluates multi-step reasoning capabilities:
     - GPT-3.5 Turbo (0-shot): 6.04% accuracy.
     - GPT-3.5 Turbo (5-shot Chain of Thought): 24.6% accuracy.
     - GPT-4 (5-shot Chain of Thought): 43.8% accuracy.
     - o1-mini (5-shot Chain of Thought): 62.3% accuracy.
   - Even with fine-tuning, GPT models struggled with HARDMath's complex reasoning tasks, highlighting the need for further advancements.

### Observations
- Fine-tuning yielded significant improvements in solving underrepresented problems like trigonometry.
- However, overall accuracy dropped, emphasizing the need for balanced datasets that cover a wide range of mathematical problems.
- For challenging datasets like HARDMath, LLMs require additional enhancements such as:
  - Incorporating logic-based tools.
  - Expanding datasets with multi-step and higher-order reasoning challenges.

### Conclusion
This study demonstrates the potential of fine-tuning LLMs to improve specific mathematical capabilities while revealing areas for further research and development. The findings underline the importance of diversified datasets and hybrid reasoning approaches for advancing mathematical problem-solving in language models.

## References

- [Measuring Mathematical Problem Solving With the MATH Dataset](https://arxiv.org/abs/2109.04990)
- [HARDMath: A Benchmark for Evaluating Mathematical Reasoning in LLMs](https://arxiv.org/abs/2304.03360)
- [ProofNet: Benchmarking Large Language Models with Formal Mathematical Proofs](https://arxiv.org/abs/2302.01995)
