# ISBI_2024_Paper: AN ENSEMBLE OF WELL-TRAINED STUDENTS CAN PERFORM ALMOST AS GOOD AS A TEACHER FOR CHEST X-RAY DIAGNOSIS

Knowledge distillation can help in generating computationally
lightweight student models for various tasks. However,
such students often show inferior performances compared
to the teacher. In this paper, we propose an ensemble
knowledge distillation approach using a group of diverse
lightweight student models for chest x-ray diagnosis. Our
novel training mechanism try to ensure that each student is
trained well by trying to be competitive with its fellow students.
We also use a feature map loss to improve the knowledge
distillation from the teacher. The ensemble of students
is significantly lightweight in terms of computation compared
to their teacher. In spite of that, experiments for multi-label
chest x-ray diagnosis show that the ensemble of well-trained
students can perform as the teacher.

## Major Highlights:
• We introduce a knowledge distillation approach with an ensemble of lightweight student models for chest x-ray diagnosis.<br>
• We present a method where the knowledge distillation is performed not only with the help of the teacher model but also taking help from the fellow student models. To do this, we introduce a novel ensemble loss..<br>
• We introduce a loss utilizing the feature maps of the teacher and each student model to help the knowledge distillation process..<br>

## Ensemble of Students
We introduce an ensemble of lightweight student models to
perform knowledge distillation. The ensemble contains four
students with two MobileNet-v2s and two ShuffleNet-v2s.
These architectures reduce the computational load, making
them ideal for resource-constrained environments. Like the
teacher model, each student model has customized FC layers.
One MobileNet-v2 has got three FC layers containing
1024, 1024, and the final classification layer (containing
neurons equal to the number of classes in the dataset), respectively.
One ShuffleNet-v2 has got similar architectural
modifications. Another MobileNet-v2 has got three FC layers
containing 1024, 512, and the final classification layer,
respectively. The other ShuffleNet-v2 has got architectural
modifications similar to the above.
<p align="center">
  <img src="images/students.jpg" alt="Ensemble model" width="500"/><br>
  <em>Fig: FC layers of students</em>
</p><br>
