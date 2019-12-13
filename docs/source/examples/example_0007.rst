example_0007_vis_curl.py
========================

The code for this example can be found here: `example_0007_vis_curl.py
<https://github.com/NREL/floris/blob/develop/examples/example_0007_vis_curl.py>`_

This example demonstrates visualizing in-plane flow using a quiver plot.  Further, to facilitate
more elaborate 3-dimensional visulizations with a program such as paraview or vapor, the flow data
is exported to VTK.  

First, to ensure the presense of in plane flows, the curl model is selected

::

        # Initialize the FLORIS interface fi
        fi = wfct.floris_interface.FlorisInterface("example_input.json")

        # Change the model to curl
        fi.floris.farm.set_wake_model('curl')



This block of code shows the methods for visualizing in-plane flows on a cut plane 5D behind the turbine

::

        # Get the vertical cut through and visualize
        cp = wfct.cut_plane.CrossPlane(fi.get_flow_data(),5*D)
        fig, ax = plt.subplots(figsize=(10,10))
        wfct.visualization.visualize_cut_plane(cp, ax=ax,minSpeed=6.0,maxSpeed=8)
        wfct.visualization.visualize_quiver(cp,ax=ax,downSamp=2)

The flow data itself is cropped for size and exported to vtk in the following block

::

        #Save the flow data as vtk
        flow_data = fi.get_flow_data()
        flow_data = flow_data.crop(flow_data,[0,20*D],[-300,300],[50,300])
        flow_data.save_as_vtk('for_3d_viz.vtk')