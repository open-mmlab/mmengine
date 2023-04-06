# Visualization

## 1 Overall Design

Visualization provides an intuitive explanation of the training and testing process of the deep learning model. In OpenMMLab, we expect the visualization module to meet the following requirements:

- Provides rich out-of-the-box features that can meet most computer vision visualization tasks.
- Versatile, expandable, and can be customized easily
- Able to visualize at anywhere in the training and testing process.
- Unified APIs for all OpenMMLab libraries, which is convenient for users to understand and use.

Based on the above requirements, we proposed the `Visualizer` and various `VisBackend` such as `LocalVisBackend`, `WandbVisBackend`, and `TensorboardVisBackend` in OpenMMLab 2.0. The visualizer could not only visualize the image data, but also things like configurations, scalars, and model structure.

- For convenience, the APIs provided by the `Visualizer` implement the drawing and storage functions.  As an internal property of `Visualizer`, `VisBackend` will be called by `Visualizer` to write data to different backends.
- Considering that you may want to write data to multiple backends after drawing, `Visualizer` can be configured with multiple backends. When the user calls the storage API of the `Visualizer`, it will traverse and call all the specified APIs of `VisBackend` internally.

The UML diagram of the two is as follows.

<div align="center">
 <img src="https://user-images.githubusercontent.com/17425982/163327736-f7cb3b16-ef07-46bc-982a-3cc7495e6c82.png" >
</div>

## 2 Visualizer

The external interface of `Visualizer` can be divided into three categories.

1. Drawing APIs

- [draw_bboxes](mmengine.visualization.Visualizer.draw_bboxes) draws a single or multiple bounding boxes
- [draw_points](mmengine.visualization.Visualizer.draw_points) draws a single or multiple points
- [draw_texts](mmengine.visualization.Visualizer.draw_texts) draws a single or multiple text boxes
- [draw_lines](mmengine.visualization.Visualizer.draw_lines) draws a single or multiple line segments
- [draw_circles](mmengine.visualization.Visualizer.draw_circles) draws a single or multiple circles
- [draw_polygons](mmengine.visualization.Visualizer.draw_polygons) draws a single or multiple polygons
- [draw_binary_masks](mmengine.visualization.Visualizer.draw_binary_masks) draws single or multiple binary masks
- [draw_featmap](mmengine.visualization.Visualizer.draw_featmap) draws feature map (**static method**)

The above APIs can be called in a chain except for `draw_featmap` because the image size may change after this method is called. To avoid confusion, `draw_featmap` is a static method.

2. Storage APIs

- [add_config](mmengine.visualization.Visualizer.add_config) writes configuration to a specific storage backend
- [add_graph](mmengine.visualization.Visualizer.add_graph) writes model graph to a specific storage backend
- [add_image](mmengine.visualization.Visualizer.add_image) writes image to a specific storage backend
- [add_scalar](mmengine.visualization.Visualizer.add_scalar) writes scalar to a specific storage backend
- [add_scalars](mmengine.visualization.Visualizer.add_scalars) writes multiple scalars to a specific storage backend at once
- [add_datasample](mmengine.visualization.Visualizer.add_datasample) the abstract interface for each repositories to draw data sample

Interfaces beginning with the `add` prefix represent storage APIs. \[datasample\] (`./data_element.md`)is the unified interface of each downstream repository in the OpenMMLab 2.0, and `add_datasample` can process the data sample directly .

3. Other APIs

- [set_image](mmengine.visualization.Visualizer.set_image) sets the original image data, the default input image format is RGB
- [get_image](mmengine.visualization.Visualizer.get_image) gets the image data in Numpy format after drawing, the default output format is RGB
- [show](mmengine.visualization.Visualizer.show) for visualization
- [get_backend](mmengine.visualization.Visualizer.get_backend) gets a specific storage backend by name
- [close](mmengine.visualization.Visualizer.close) closes all resources, including `VisBackend`

For more details, you can refer to [Visualizer Tutorial](../advanced_tutorials/visualization.md).

## 3 VisBackend

After drawing, the drawn data can be stored in multiple visualization storage backends. To unify the interfaces, MMEngine provides an abstract class, `BaseVisBackend`, and some commonly used backends such as `LocalVisBackend`, `WandbVisBackend`, and `TensorboardVisBackend`.
The main interfaces and properties of `BaseVisBackend` are as follows:

- [add_config](mmengine.visualization.BaseVisBackend.add_config) writes configuration to a specific storage backend
- [add_graph](mmengine.visualization.BaseVisBackend.add_graph) writes model graph to a specific backend
- [add_image](mmengine.visualization.BaseVisBackend.add_image) writes image to a specific backend
- [add_scalar](mmengine.visualization.BaseVisBackend.add_scalar) writes scalar to a specific backend
- [add_scalars](mmengine.visualization.BaseVisBackend.add_scalars) writes multiple scalars to a specific backend at once
- [close](mmengine.visualization.BaseVisBackend.close) closes the resource that has been opened
- [experiment](mmengine.visualization.BaseVisBackend.experiment) writes backend objects, such as WandB objects and Tensorboard objects

`BaseVisBackend` defines five common data writing interfaces. Some writing backends are very powerful, such as WandB, which could write tables and videos. Users can directly obtain the `experiment` object for such needs and then call native APIs of the corresponding backend. `LocalVisBackend`, `WandbVisBackend`, and `TensorboardVisBackend` are all inherited from `BaseVisBackend` and implement corresponding storage functions according to their features. Users can also customize `BaseVisBackend` to extend the storage backends and implement custom storage requirements.

For more details, you can refer to [Storage Backend Tutorial](../advanced_tutorials//visualization.md).
