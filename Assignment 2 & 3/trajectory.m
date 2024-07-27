% source - https://in.mathworks.com/help/robotics/ug/simulate-different-kinematic-models-for-mobile-robots.html

num_iter = 3;
paths = [];
for i = 1:num_iter
    bicycle = bicycleKinematics(VehicleInputs="VehicleSpeedHeadingRate",MaxSteeringAngle=pi);
    waypoints = load("path_"+i+".mat").path';

    sampleTime = 0.05;
    finalTime = 1e3;
    tVec = 0:sampleTime:finalTime;
    % taking angle between the first 2 points to map the inital angle
    theta = atan2(waypoints(2, 2) - waypoints(1,2), waypoints(2, 1) - waypoints(1,1));
    initPose = [waypoints(1,:)'; theta];
    
    scalingFactor = 1;
    controller = controllerPurePursuit(Waypoints=waypoints,DesiredLinearVelocity=scalingFactor,MaxAngularVelocity=scalingFactor*pi);

    goalPoints = waypoints(end,:)';
    goalRadius = 0.5;
    
    [tBicycle,bicyclePose] = ode45(@(t,y)derivative(bicycle,y,exampleHelperMobileRobotController(controller,y,goalPoints,goalRadius)),tVec,initPose);
    bicycleTranslations = [bicyclePose(:,1:2) zeros(length(bicyclePose),1)];
    bicycleRot = axang2quat([repmat([0 0 1],length(bicyclePose),1) bicyclePose(:,3)]);
    
    figure
    plot(waypoints(:,1),waypoints(:,2),"kx-",MarkerSize=20);
    hold all
    sampleTimePlot = 25;
    plotTransforms(bicycleTranslations(1:sampleTimePlot:end,:),bicycleRot(1:sampleTimePlot:end,:),MeshFilePath="groundvehicle.stl",MeshColor="r");
    
    obstacles = [2, 4.5, 3; 2, 3, 12; 3, 15, 15];
    for counter = 1:length(obstacles)
        angle = linspace(0, 2*pi, 100);
        radius = obstacles(counter, 1);
        center = [obstacles(counter, 2), obstacles(counter, 3)];
        x = center(1) + radius * cos(angle);
        y = center(2) + radius * sin(angle);
        plot(x, y, 'k', 'LineWidth', 2);
    end
    axis equal
    view(0,90)
end



