# Aims

The project's aims are to collect track and event data from football footage. The ultimate goal is to be able to process any kind of footage but, for now, it will only process broadcast football matches.


# Objectives

+ detect humans
    + filter out referee(s)
    + detect players
    + identify players numbers
    + determine players position
+ detect ball
    + determine its position
    + track its position
+ detect pitch
    + detect its geometry
    + detect corner flags
    + detect goals
+ collect data
    + collect event data
    + collect tracking data
    + synchronise event and tracking data



# Techniques

+ Image Processing
+ Object detection (YOLO v4)
+ Object interaction tracking
+ Object motion tracking
+ Multi-algorithm implementation (Detection -> tracking -> identification)
+ Machine learning (Extrapolation; training against current data)
+ Human pose estimation
+ grid positioning
+ 3D human interaction

# Challenges

+ synchronise event and tracking data
+ geometry reconstruction
+ optical distortion
+ player's numbers extraction
+ predict possible position of off sight players
+ weather and light visual conditions
+ different camera/editing/perspective
+ action detection
+ positioning flipping
+ 3D ball positioning estimation using physics model
+ low quality footage
+ deal with processing power
+ synchronise footage and data timestamps


# Limitations

- Human agent must verify and validate data collection
- Human agent must input match meta data
- Human agent must supervise/calibrate video processing
- Tracking broadcast is affected by zoom/replays and camera changes

# To dos

+ sign up for google jupyter notebook codelabs
+ set up public database for academics
+ when downloading copyrighted recordings or process data, reference source and its copyright
+ save collected data in public database


# Research materials and resources

[https://sam.johnson.io/research/lsp.html](https://sam.johnson.io/research/lsp.html)

[https://posetrack.net/](https://posetrack.net/)

[https://www.vislab.ucr.edu/PUBLICATIONS/pubs/Journal%20and%20Conference%20Papers/after10-1-1997/Conference/2018/FINAL-published-soccer-ball-generating.pdf](https://www.vislab.ucr.edu/PUBLICATIONS/pubs/Journal%20and%20Conference%20Papers/after10-1-1997/Conference/2018/FINAL-published-soccer-ball-generating.pdf)

[https://paperswithcode.com/dataset/kth-multiview-football-i](https://paperswithcode.com/dataset/kth-multiview-football-i)

[https://towardsdatascience.com/football-games-analysis-from-video-stream-with-machine-learning-745e62b36295](https://towardsdatascience.com/football-games-analysis-from-video-stream-with-machine-learning-745e62b36295)

[https://neptune.ai/blog/dive-into-football-analytics-with-tensorflow-object-detection-api](https://neptune.ai/blog/dive-into-football-analytics-with-tensorflow-object-detection-api)

[https://paperswithcode.com/dataset/kth-multiview-football-i](https://paperswithcode.com/dataset/kth-multiview-football-i)

[https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w34/Theagarajan_Soccer_Who_Has_CVPR_2018_paper.pdf](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w34/Theagarajan_Soccer_Who_Has_CVPR_2018_paper.pdf)

[https://arxiv.org/pdf/1810.10658.pdf](https://arxiv.org/pdf/1810.10658.pdf)

[https://www.csc.kth.se/cvap/cvg/?page=footballdataset2](https://www.csc.kth.se/cvap/cvg/?page=footballdataset2)

[https://www.researchgate.net/figure/Football-pitch-with-dimensions-of-bounds_fig1_292995807](https://www.researchgate.net/figure/Football-pitch-with-dimensions-of-bounds_fig1_292995807)

[https://www.youtube.com/watch?v=hs_v3dv6OUI](https://www.youtube.com/watch?v=hs_v3dv6OUI)

[https://www.youtube.com/watch?v=jLEv14GAcbs](https://www.youtube.com/watch?v=jLEv14GAcbs)

[https://www.youtube.com/watch?v=Iu3V8hTnlVg](https://www.youtube.com/watch?v=Iu3V8hTnlVg)

[https://github.com/paperswithcode/releasing-research-code](https://github.com/paperswithcode/releasing-research-code)

[https://dev.to/stephan007/open-source-sports-video-analysis-using-maching-learning-2ag4](https://dev.to/stephan007/open-source-sports-video-analysis-using-maching-learning-2ag4)

[https://spiral.imperial.ac.uk/bitstream/10044/1/12702/4/DeardenDemirisGrauCVMP06.pdf](https://spiral.imperial.ac.uk/bitstream/10044/1/12702/4/DeardenDemirisGrauCVMP06.pdf)

[https://www.sciencedirect.com/science/article/pii/S0045790620305541#fig0006](https://www.sciencedirect.com/science/article/pii/S0045790620305541#fig0006)

[https://www.mdpi.com/2076-3417/10/1/24/htm](https://www.mdpi.com/2076-3417/10/1/24/htm)

[https://towardsdatascience.com/is-it-possible-to-stop-messi-a-data-perspective-cf4e2d900181](https://towardsdatascience.com/is-it-possible-to-stop-messi-a-data-perspective-cf4e2d900181)

[https://github.com/google-research/football](https://github.com/google-research/football)

[https://github.com/vcg-uvic/sportsfield_release/issues/5](https://github.com/vcg-uvic/sportsfield_release/issues/5)

[https://github.com/vcg-uvic/sportsfield_release](https://github.com/vcg-uvic/sportsfield_release)

[https://deepmind.com/blog/article/advancing-sports-analytics-through-ai](https://deepmind.com/blog/article/advancing-sports-analytics-through-ai)

[https://colab.research.google.com/](https://colab.research.google.com/)

[https://notebook.xbdev.net/index.php?page=bodypix&](https://notebook.xbdev.net/index.php?page=bodypix&)

