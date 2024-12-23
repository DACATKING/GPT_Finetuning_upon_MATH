# GPT_Finetuning_upon_MATH

## Description

This project focuses on fine-tuning GPT-3.5 and GPT-4 models to enhance their mathematical problem-solving capabilities. By leveraging specialized datasets and training scripts, the aim is to improve the models' proficiency in understanding and solving mathematical problems.

## Repository Structure

- `fine_tune_gpt3.5_math_main.py`: Main script for fine-tuning the GPT-3.5 model on mathematical datasets.
- `fine_tune_gpt4omini_math_main.py`: Main script for fine-tuning the GPT-4 model on mathematical datasets.
- `README.md`: Provides an overview of the project and instructions for usage.

## Usage

To fine-tune the models, follow these steps:

   - For GPT-3.5:
     ```bash
     python fine_tune_gpt3.5_math_main.py --data_path=processed_data.json --output_dir=gpt3.5_finetuned
     ```
   - For GPT-4:
     ```bash
     python fine_tune_gpt4omini_math_main.py --data_path=processed_data.json --output_dir=gpt4_finetuned
     ```

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
