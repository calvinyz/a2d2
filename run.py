from sys_controller.process import SysProcController

if __name__ == '__main__':
    proc_ctr = SysProcController()
    proc_ctr.run()

    # Note that simGetImage will freeze the window and that will make the video not smooth.
    # Connect DRL when fire detected
    # Show drone image (with gcc border box and fire detection bounding box?)
    # Implement alert with visual

    # -> Train wildfire detection model
    # Train with FLAME dataset
    # Retrain with Airsim dataset   

    # -> Train DRL model
    
    # Re-collect image dataset
    #   - Write a script to set a singel fixed fire and move drone to different positions and heights
    #   - Annotate the dataset

    # * (Later) Update grid search code
    # *   - With the retrained model, run grid search to find optimal drone height and camera angle

    # Fix patrol path, make it slower (velocity) and smoother 
    #   - Fix camera direction?
    #   - Add fire detection during patrol (How?)
    #   - Change from patrol to start inspection?

    # Inspection path
    #   - Run trained DRL next step

    # Implement a (simpler) DRL model
    #   - DQN?
    #   - More? 

    # Implement alert
    #  - Threshold 

