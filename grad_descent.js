/*
Interactive gradient descent explorer
Author: Kyle Bradbury
Idea adapted from: http://bl.ocks.org/WilliamQLiu/76ae20060e19bf42d774
*/

// TODO LIST FOR PROJECT:
// [X] Button to generate new data
// [X] Slider to adjust number of data points
// [X] Can hover over a point along the Error plot and display the hypothesis function
// [X] Can click on the error plot to lock in point
// [X] Can move the training data points
// [X] Update dot when sliders or button pressed
// [X] Create slider to adjust noise level
// [X] Color the points according to model to judge fit
// [X] Calculate gradient
// [X] Move in the direction of gradient (batch)
// [X] Take step with SGD
// [X] Checkbox to show/hide target model
// [X] Plot the learning curves by epoch
// [X] Assign the average value of error through the epoch to each epoch
// [X] Set max number of datapoints to 300
// [X] When performing SGD, show the points that are selected as part of the batch
//     and the corresponding error function just for that subset
// [X] Button to turn on plot for the hypothesis function
// [X] Add target line at true parameter value
// [X] Highlight which specific points were used in the SGD update
// [X] Toggle for SGD batch error function plotting

// [ ] Change the target function
// [ ] Change the target function
// [ ] Add legend
// [ ] Checkbox to color the dots (or not)
// [ ] Make learning rate decay a function of epoch
// [ ] Show the last point for the SGD update to be able to see the progress more easily
// [ ] Add convergence criterion based on last 10 batches (if the change in x < delta for 10 consecutive batches)
// [ ] Update minibatch error when a value is moved
// [ ] Fix toggle for target function (function automatically reappears when new data is drawn)

// May be unnecessary:
// [ ] Checkbox to show/hide target error point
// [ ] Slider to adjust target weight
// [ ] Can delete the training data points

/*
--------------------------------------------
CODE FOR COMPUTATION
--------------------------------------------
*/
function linspace(startValue, stopValue, cardinality) {
  var arr = [];
  var step = (stopValue - startValue) / (cardinality - 1);
  for (var i = 0; i < cardinality; i++) {
    arr.push(startValue + (step * i));
  }
  return arr;
}

function mse(weights,data) {
    let N = data.length;
    let err = weights.map(function(w){
        return sum_squared_error(data,f_generator(w)) / N ;
    })
    return make_d3_ready({x:weights, y:err}) ;
}

function get_current_error(weight,data) {
    let N = data.length;
    return sum_squared_error(data,f_generator(weight)) / N ;
}

function save_current_error() {
    err = get_current_error(w_hat_old,state.data);
    error_within_epoch.push(err);
}

function sum_squared_error(data,f) {
    return data.reduce(function(tot,d){
        return tot + Math.pow(f(d.x) - d.y,2) ;
    }, 0)
}

function square_error(data,f) {
    return data.map(function(d){
        return Math.pow(f(d.x) - d.y,2) ;
    })
}

function f_generator(w) {
    return function(x) {
        return Math.sin(w*x) ; 
    }
}

function f_grad(d,w) {
    return (Math.sin(w*d.x) - d.y)*Math.cos(w*d.x)*d.x;
    // return (w*d.x-1.5)*d.x
}

function rand() {
    return Math.random();
}

function randn() {
    var u = 0, v = 0;
    while(u === 0) u = Math.random(); //Converting [0,1) to (0,1)
    while(v === 0) v = Math.random();
    return Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
}

function rand_array(maxval,N) {
    let rand_vals = [];
    for (let i=0; i < N; i++) {
        rand_vals.push(rand()*maxval);
    }
    return rand_vals;
}

// Create a representation of the true function at equally spaced x values
function gen_function_data(f,n,xmax) {
    let x = linspace(0,xmax,n) ;
    let y = x.map(v => f(v)) ;
    return make_d3_ready({x:x, y:y}) ;
}

// Create the noise-free data but with random values for x
function gen_function_data_random_x(f,N,xmax) {
    let x = rand_array(xmax,N) ;
    x.sort();
    // let x = linspace(0,xmax,N) ;
    let y = x.map(v => f(v)) ;
    
    return make_d3_ready({x:x, y:y}) ;
}

// Generate an array of normally distributed noise
function gen_noise(N) {
    let noise = [];
    for (let i = 0; i < N; i++) {
        noise.push(randn());
    }
    return noise ;
}

// Add noise to "clean" data
function corrupt_data(raw_function,noise) {
    let output = [];
    N = raw_function.length;
    for (let i = 0; i < N; i++) {
        output.push({x:raw_function[i].x, y:raw_function[i].y + noise_std*noise[i]}) ;
    }
    return output 
}

// Convert the arrays into arrays of objects with (x,y) pairs: [{x:x1, y:y1},...]
function make_d3_ready(data) {
    let d3data = []; 
    for (let i = 0; i < data.x.length; i++) {
        d3data.push({x:data.x[i], y:data.y[i]}) ;
    }
    return d3data;
}

// Calculate the gradient of the loss function based on the training data and f_generator
// Assumes the case where f(w,x) = sin(w*x)
// Here data is an array of (x,y) pair objects
function gradient() {
    let w = w_hat,
        grad = 0,
        cgrad = [],
        d = [];
    
    batch_indices = get_next_batch();
    let N = batch_indices.length;

    batch_indices.forEach(function(i){
        d = state.data[i];
        grad = grad + f_grad(d,w);
    })
    full_gradient = (2 / N) * grad;
    return full_gradient;
}

function get_next_batch() {
    let N = state.data.length ;
    if (sgd_mode == 'batch') {
        // Return a list of all the indices
        return get_range(N);
    } else if (sgd_mode == 'minibatch') {
        // Check if the pool is empty: if so, refill it
        return draw_indices(batchsize) ;
    }
}

// Get values from 0 to N-1 by intervals of 1
function get_range(N) {
    let list = [];
    for (let i = 0; i < N; i++) {
        list.push(i);
    }
    return list;
}

// Pick a new set of training data indices for the next batch
function draw_indices(N) {
    let batch = [];
    for (let i = 0; i < N; i++) {
        // If there are no more indices, refill them
        let nAvailableIndices = indices_sgd.length;
        if (nAvailableIndices == 0) {
            indices_sgd = get_range(state.data.length);
            nAvailableIndices = indices_sgd.length;
            new_epoch() ;
        }
        // Draw one value and remove it from the index
        let randomIndex = Math.floor(Math.random()*nAvailableIndices); 
        cindex = indices_sgd.splice(randomIndex,1)[0];
        batch.push(cindex);
    }
    return batch;
}

function new_epoch() {
    epoch++;
    // console.log(error_within_epoch)
    if (epoch <= EPOCH_MAX) {
        let epoch_mean_error = mean(error_within_epoch) ;
        error_by_epoch.push({x:epoch-1,y:epoch_mean_error});
        error_within_epoch = [];
        update_learning_curve();
    }
}

function mean(x) {
    sumx = x.reduce(function(acc,val){
        return acc + val;
    },0)
    return sumx / x.length
}

function sgd_update() {
    // Save the error to an array for learning curves
    save_current_error()

    // Get the gradient
    let grad = gradient() ;

    // Calculate new value for w
    w_hat_old = w_hat;
    w_hat = w_hat - learning_rate * grad;

    // Keep the selection within bounds on the plot
    if (w_hat < WMIN) {
        w_hat = WMIN ;
    } else if (w_hat > WMAX) {
        w_hat = WMAX ;
    }

    // Reduce the learning rate over time
    learning_rate = 0.9999*learning_rate;
}

function get_batch_error() {
    data = get_batch_data();
    batch_error = mse(state.weights,data);
}

function get_batch_data() {
    data = [];
    batch_indices.forEach(function(i) {
        data.push(state.data[i])
    })
    return data;
}

// Define variables that will not be changed interactively
let XMAX = Math.PI,
    YMIN = -1.5, 
    YMAX = 1.5,
    WMIN = 0,
    WMAX = 30,
    EMIN = 0,
    EMAX = 2.5,
    NMAX = 300,
    EPOCH_MIN = 0,
    EPOCH_MAX = 400,
    nTarget = 601,
    nWeights = 601,
    true_weight = 5,
    NOISE_MAX = 0.5,
    LR_MAX = 1,
    f = f_generator(true_weight);

// Define parameters that may change interactively
let nData = 100,
    w_hat = 10,
    w_hat_old = 10,
    selected_weight_index = [],
    noise_std = 0.25,
    hypothesis_locked = false,
    indices_sgd = [],
    sgd_mode = 'minibatch',
    batchsize = 4,
    learning_rate = 0.5,
    sgd_timer_interval = [],
    interval_set = false,
    epoch = 0,
    full_gradient = 0
    batch_indices = [],
    batch_error = [],
    show_target_function = true,
    show_batch_error = true,
    show_gradient_line = true,
    show_learning_curve = true,
    error_by_epoch = [],
    error_within_epoch = [];

// Parameters for plots
var w = 400,
    h = 400,
    margin = { top: 40, right: 20, bottom: 30, left: 40 };
let radius = 5;

// Create a new set of all values for the data
function new_scenario() {
    let target = gen_function_data(f,nTarget,XMAX),
        noisefree_data = gen_function_data_random_x(f,nData,XMAX),
        noise = gen_noise(nData),
        data = corrupt_data(noisefree_data, noise);
        weights = linspace(WMIN,WMAX,nWeights),
        error_data = mse(weights,data),
        pointwise_error = square_error(data, f_generator(w_hat))
        hypothesis = [] ;
        if (w_hat !== '') {
            hypothesis = gen_function_data(f_generator(w_hat),nTarget,XMAX)  
        }
        
    return {target:target, 
            noisefree_data:noisefree_data,
            noise:noise,
            data:data, 
            weights:weights, 
            error_data:error_data,
            pointwise_error:pointwise_error,
            hypothesis:hypothesis};
}

function reset_sgd() {
    learning_rate = document.getElementById("slideLearningRate").value / 100 * LR_MAX;
    epoch = 0;
    error_by_epoch = [];
    error_within_epoch = [];
    update_learning_curve();
    document.getElementById('learningrate').innerHTML = "Learning Rate = " + learning_rate.toFixed(4) ;
    document.getElementById('weight').innerHTML = "Weight = " + w_hat.toFixed(3) ;
    document.getElementById('epoch').innerHTML = "Epoch = " + epoch ;
    
}

function recalculate_error() {
    recalculate_pointwise_error();
    state.error_data = mse(state.weights,state.data);
}

function recalculate_noise() {
    state.data = corrupt_data(state.noisefree_data, state.noise);
    state.error_data = mse(state.weights,state.data)
}

function recalculate_hypothesis() {
    state.hypothesis = gen_function_data(f_generator(w_hat),nTarget,XMAX)  
}

function recalculate_pointwise_error() {
    state.pointwise_error = square_error(state.data, f_generator(w_hat))
}

// Draw new data and recalculate component values
function refresh() {
    state = new_scenario();
    update_error_line();
    update_target_line();
    update_hypothesis_line();
    update_training_circles();  
}

/*
--------------------------------------------
PLOT DATA, TARGET, AND ESTIMATE
--------------------------------------------
*/

// We're passing in a function in d3.max to tell it what we're maxing (x value)
var xScale = d3.scaleLinear()
    .domain([0, XMAX])
    .range([margin.left, w - margin.right]);  // Set margins for x specific

// We're passing in a function in d3.max to tell it what we're maxing (y value)
var yScale = d3.scaleLinear()
    .domain([YMIN, YMAX])
    .range([h - margin.bottom, margin.top]);  // Set margins for y specific

// Add a X and Y Axis (Note: orient means the direction that ticks go, not position)
var xAxis = d3.axisBottom(xScale);
var yAxis = d3.axisLeft(yScale);

// Define the line
var valueline = d3.line()
    .x(function(d) { return xScale(d.x); })
    .y(function(d) { return yScale(d.y); });

// Define the color scale
var error_color_scale = d3.scaleLinear()
    .domain([0,2])
    .range(["orange", "black"])

var svg = d3.select("#plot_function")
    .append("svg")
    .attr("width", w)
    .attr("height", h);

// Adds X-Axis as a 'g' element
svg.append("g").attr("class", "axis")
  .attr("transform", "translate(" + [0, h - margin.bottom] + ")")  // Translate just moves it down into position (or will be on top)
  .call(d3.axisBottom(xScale));  // Call the xAxis function on the group

// Adds Y-Axis as a 'g' element
svg.append("g").attr("class", "axis")
  .attr("transform", "translate(" + [margin.left, 0] + ")")
  .call(yAxis);  // Call the yAxis function on the group

// Add axis labels
svg.append("text")
    .attr("class", "x label")
    .attr("text-anchor", "end")
    .attr("alignment-baseline", "bottom")
    .attr("x", w)
    .attr("y", h)
    .text("x");

svg.append("text")
    .attr("class", "y label")
    .attr("text-anchor", "middle")
    .attr("y", 6)
    .attr("dy", ".75em")
    .attr("x", -margin.top)
    .attr("transform", "rotate(-90)")
    .text("f(x,w)");


function update_target_line() {
    svg.selectAll("path.targetline").remove()

    svg.append("path")
        .attr("class", "targetline")
        .attr("id", "target")
        .attr("d", valueline(state.target));
}

function update_hypothesis_line() {
    svg.selectAll("path.hypothesisline").remove()

    svg.append("path")
        .attr("class", "hypothesisline")
        .attr("id", "hypothesis")
        .attr("d", valueline(state.hypothesis));
}

function update_training_circles() {
    var training_circles = svg.selectAll("circle")
    .data(state.data) ;

    training_circles.exit().remove(); 
    training_circles.enter()
        .append("circle")
        .merge(training_circles)
        .attr("cx", function(d) { return xScale(d.x); })
        .attr("cy", function(d) { return yScale(d.y); })
        .attr("r", radius)
        .attr("index", function(d,i) {return i;})
        .attr("fill", function(d,i){ 
            return error_color_scale(state.pointwise_error[i]); 
        })
        .attr('class', function(d,i) {
            if (batch_indices.includes(i)) {
                return "batch-sample";
            } else {
                return "";
            }
        })
        .call(d3.drag()
                  .on("start", dragstarted)
                  .on("drag", dragged)
                  .on("end", dragended)
                  )
        .raise();
}

/*
--------------------------------------------
DRAG TO CHANGE DATA
--------------------------------------------
*/

function dragstarted(d) {
    let index = d3.select(this).attr("index") ;
    d.x = d3.event.x;
    d.y = d3.event.y;
    d3.select(this).attr("class", "dragging")
        .attr("r", radius * 1.5)
        .raise();
}

function dragged(d,i) {
    // Use d3.mouse instead of d3.event.x to work around issues with scaling the data
    var coords = d3.mouse(this);
    let index = d3.select(this).attr("index") ;

    // Update the datapoint to match the new position
    d.x = xScale.invert(coords[0]);
    d.y = yScale.invert(coords[1]);

    // Check to make sure points don't move off the plot
    if (d.x < 0) {d.x = 0;}
    if (d.x > XMAX) {d.x = XMAX;}
    if (d.y < YMIN) {d.y = YMIN;}
    if (d.y > YMAX) {d.y = YMAX;}

    // Redraw the error functions based on the new data
    recalculate_error();
    update_error_line();

    d3.select(this)
        .attr("cx", xScale(d.x))
        .attr("cy", yScale(d.y))
        .attr("r", radius * 1.5)
        .attr("fill", function(d,i){ 
            return error_color_scale(state.pointwise_error[index]); 
        })
        .raise();
}

function dragended(d) {
    d3.select(this).attr("class", 'dragged')
        .attr("r", radius);
}

/*
--------------------------------------------
PLOT ERROR
--------------------------------------------
*/

// We're passing in a function in d3.max to tell it what we're maxing (x value)
var xScale_ep = d3.scaleLinear()
    .domain([WMIN, WMAX])
    .range([margin.left, w - margin.right]);  // Set margins for x specific

// We're passing in a function in d3.max to tell it what we're maxing (y value)
var yScale_ep = d3.scaleLinear()
    .domain([EMIN, EMAX])
    .range([h - margin.bottom, margin.top]);  // Set margins for y specific

// Add a X and Y Axis (Note: orient means the direction that ticks go, not position)
var xAxis_ep = d3.axisBottom(xScale_ep);
var yAxis_ep = d3.axisLeft(yScale_ep);

// Define the line
var valueline_ep = d3.line()
    .x(function(d) { return xScale_ep(d.x); })
    .y(function(d) { return yScale_ep(d.y); });

var svg_ep = d3.select("#plot_error")
    .append("svg")
    .attr("width", w)
    .attr("height", h);

// Adds X-Axis as a 'g' element
svg_ep.append("g").attr("class", "axis")
  .attr("transform", "translate(" + [0, h - margin.bottom] + ")")  // Translate just moves it down into position (or will be on top)
  .call(d3.axisBottom(xScale_ep));  // Call the xAxis function on the group

// Adds Y-Axis as a 'g' element
svg_ep.append("g").attr("class", "axis")
  .attr("transform", "translate(" + [margin.left, 0] + ")")
  .call(yAxis_ep);  // Call the yAxis function on the group

// Add axis labels
svg_ep.append("text")
    .attr("class", "x label")
    .attr("text-anchor", "end")
    .attr("alignment-baseline", "bottom")
    .attr("x", w)
    .attr("y", h)
    .text("w");

svg_ep.append("text")
    .attr("class", "y label")
    .attr("text-anchor", "middle")
    .attr("y", 6)
    .attr("dy", ".75em")
    .attr("x", -margin.top)
    .attr("transform", "rotate(-90)")
    .text("E(w)");

svg_ep.append("rect")
        .attr("class", "rect-for-mouseover")
        .attr("width", w)
        .attr("height", h)
        .on("mousemove", mousemoved)
        .on("click", error_plot_clicked);

svg_ep.append("path")
        .attr("class", "true-weight-line")
        .attr("d", valueline_ep([{x:true_weight, y:EMIN},{x:true_weight, y:EMAX}]));

function update_error_line() {
    svg_ep.selectAll("path.line").remove()

    svg_ep.append("path")
        .attr("class", "line")
        .attr("d", valueline_ep(state.error_data));

    update_error_point()
}

function update_batch_error_line() {
    svg_ep.selectAll("path.batch-error-line").remove()

    if (show_batch_error) {
        svg_ep.append("path")
            .attr("class", "batch-error-line")
            .attr("d", valueline_ep(batch_error));
    }
}

// Create a single circle for highlighting selected point
old_point = svg_ep.append("circle")
    .attr("id","old-error_point");

svg_ep.append("circle")
    .attr("id","error_point")
    .on("click", error_plot_clicked);

svg_ep.append("path")
        .attr("class", "gradient")
        .attr("d", valueline_ep([
            {x:w_hat_old, y:yScale_ep.invert(d3.select('#error_point').attr('cy'))},
            {x:w_hat, y:yScale_ep.invert(d3.select('#error_point').attr('cy'))}]));

function update_gradient_line() {
    svg_ep.selectAll("path.gradient").remove()

    if (show_gradient_line) {
        svg_ep.append("path")
            .attr("class", "gradient")
            .attr("d", valueline_ep([
                {x:w_hat_old, y:yScale_ep.invert(d3.select('#error_point').attr('cy'))},
                {x:w_hat, y:yScale_ep.invert(d3.select('#error_point').attr('cy'))}]));
    }
}

/*
--------------------------------------------
MOUSE HOVER INTERACTION FOR ERROR PLOT
--------------------------------------------
*/

function error_plot_clicked() {
    hypothesis_locked = !hypothesis_locked;
}

function mousemoved() {
    if (hypothesis_locked) {return;}
    var coord = d3.mouse(this);
    w_hat = xScale_ep.invert(coord[0]);
    if (w_hat < WMIN) {
        w_hat = WMIN ;
    } else if (w_hat > WMAX) {
        w_hat = WMAX ;
    }

    selected_weight_index = get_nearest_x(w_hat);

    // remove_existing_highlight();
    select_weight(selected_weight_index);

    recalculate_hypothesis();
    update_hypothesis_line();
    recalculate_pointwise_error();
    update_training_circles();
}

function select_weight(index) {
    let d = state.error_data[index];
    // w_hat = state.weights[index];

    d3.select('#error_point')
        .attr("fill", "orange")
        .attr("r", radius)
        .attr("cx", xScale_ep(w_hat))
        .attr("cy", yScale_ep(d.y))
        .attr('weight_index',index)
        .raise();
    
}

function update_error_point() {
    let d = state.error_data[selected_weight_index];
    d3.select('#error_point')
        .attr("cx", xScale_ep(d.x))
        .attr("cy", yScale_ep(d.y))
        .raise();
}

function get_nearest_x(x) {
    var mindist = 10e6;
    var dist = [];
    var index = [];
    state.error_data.forEach(function(d,i) {
        dist = Math.abs(d.x - x);
        if (dist < mindist) {
            index = i;
            mindist = dist ;
        }
    })
    return index;
}

function update_for_sgd() {
    sgd_update()
    recalculate_hypothesis();
    update_hypothesis_line();
    recalculate_pointwise_error();
    
    selected_weight_index = get_nearest_x(w_hat);
    select_weight(selected_weight_index);
    document.getElementById('learningrate').innerHTML = "Learning Rate = " + learning_rate.toFixed(4) ;
    document.getElementById('weight').innerHTML = "Weight = " + w_hat.toFixed(3) ;
    document.getElementById('epoch').innerHTML = "Epoch = " + epoch ;

    
    if (show_batch_error) {
        get_batch_error();    
    }
    
    update_batch_error_line();
    
    update_training_circles();

    update_gradient_line();
}


function autoupdate_for_sgd() {
    if (interval_set == true) {
        window.clearInterval(sgd_timer_interval);
        interval_set = false;
    } else {
        sgd_timer_interval = window.setInterval(update_for_sgd,0) ;    
        interval_set = true;
    }
    
}

function toggle_sgd_mode() {
    let buttonlabel = document.getElementById("sgd_mode_toggle");
    if (sgd_mode == 'batch') {
        sgd_mode = 'minibatch';
        buttonlabel.innerHTML = 'Toggle Mode: ' + sgd_mode ;
    } else if (sgd_mode == 'minibatch') {
        sgd_mode = 'batch';
        buttonlabel.innerHTML = 'Toggle Mode: ' + sgd_mode ;
    }
}

function toggle_target() {
    if (show_target_function) {
        d3.select('#target').attr('visibility','hidden')
    } else {
        d3.select('#target').attr('visibility','visible')
    }
    show_target_function = !show_target_function;
}

function toggle_batch_error() {
    if (!show_batch_error) {
        get_batch_error()
    } 
    show_batch_error = !show_batch_error;
    update_batch_error_line();
}

function toggle_gradient() {
    show_gradient_line = !show_gradient_line;
    update_gradient_line();
}


/*
--------------------------------------------
LEARNING CURVE PLOT
--------------------------------------------
*/
let h_lc = h/2;
// We're passing in a function in d3.max to tell it what we're maxing (x value)
var xScale_lc = d3.scaleLinear()
    .domain([EPOCH_MIN,EPOCH_MAX])
    .range([margin.left, w - margin.right]);  // Set margins for x specific

// We're passing in a function in d3.max to tell it what we're maxing (y value)
var yScale_lc = d3.scaleLinear()
    .domain([EMIN, EMAX])
    .range([h_lc - margin.bottom, margin.top]);  // Set margins for y specific

// Add a X and Y Axis (Note: orient means the direction that ticks go, not position)
var xAxis_lc = d3.axisBottom(xScale_lc);
var yAxis_lc = d3.axisLeft(yScale_lc);

// Define the line
var valueline_lc = d3.line()
    .x(function(d) { return xScale_lc(d.x); })
    .y(function(d) { return yScale_lc(d.y); });

var svg_lc = d3.select("#plot_learning_curve")
    .append("svg")
    .attr("width", w)
    .attr("height", h_lc);

// Adds X-Axis as a 'g' element
svg_lc.append("g").attr("class", "axis")
  .attr("transform", "translate(" + [0, h_lc - margin.bottom] + ")")  // Translate just moves it down into position (or will be on top)
  .call(d3.axisBottom(xScale_lc));  // Call the xAxis function on the group

// Adds Y-Axis as a 'g' element
svg_lc.append("g").attr("class", "axis")
  .attr("transform", "translate(" + [margin.left, 0] + ")")
  .call(yAxis_lc);  // Call the yAxis function on the group

// Add axis labels
svg_lc.append("text")
    .attr("class", "x label")
    .attr("text-anchor", "end")
    .attr("alignment-baseline", "bottom")
    .attr("x", w)
    .attr("y", h_lc-5)
    .text("epoch");

svg_lc.append("text")
    .attr("class", "y label")
    .attr("text-anchor", "end")
    .attr("y", 6)
    .attr("dy", ".75em")
    .attr("x", -margin.top)
    .attr("transform", "rotate(-90)")
    .text("Mean E(w) in Epoch");


function update_learning_curve() {
    svg_lc.selectAll("path.learning-curve").remove()

    if (show_learning_curve) {
        svg_lc.append("path")
            .attr("class", "learning-curve")
            .attr("d", valueline_lc(error_by_epoch));
    }
}

/*
--------------------------------------------
NUMBER OF TRAINING POINTS CONTROL SLIDER
--------------------------------------------
*/
// Note: since the slider is restricted to whole numbers, we have to convert from 0 to 100:

var sliderNumSamples = document.getElementById("slideNumSamples");
var numsamples_slider = document.getElementById("slideNumSamplesLabel");
sliderNumSamples.value = Math.round(nData / NMAX * 100); 
numsamples_slider.innerHTML = "Number of training datapoints = " + nData; // Display the default slider value

// Update the current slider value (each time you drag the slider handle)
sliderNumSamples.oninput = function() {
    nData = Math.round(this.value /100 * NMAX);
    numsamples_slider.innerHTML = "Number of training datapoints = " + nData;
    indices_sgd = []
    refresh();
}

/*
--------------------------------------------
NOISE CONTROL SLIDER
--------------------------------------------
*/
// Note: since the slider is restricted to whole numbers, we have to convert from 0 to 100:

var sliderNoise = document.getElementById("slideNoise");
var noise_slider = document.getElementById("slideNoiseLabel");
sliderNoise.value = Math.round(noise_std / NOISE_MAX * 100); 
noise_slider.innerHTML = "Noise std = " + noise_std; // Display the default slider value

// Update the current slider value (each time you drag the slider handle)
sliderNoise.oninput = function() {
    noise_std = this.value /100 * NOISE_MAX;
    noise_slider.innerHTML = "Noise std = " + noise_std;

    // Update the plot
    recalculate_noise()
    update_training_circles();
    update_error_line();
}

/*
--------------------------------------------
LEARNING RATE CONTROL SLIDER
--------------------------------------------
*/
// Note: since the slider is restricted to whole numbers, we have to convert from 0 to 100:

var sliderLearningRate = document.getElementById("slideLearningRate");
var learning_rate_slider = document.getElementById("slideLearningRateLabel");
sliderLearningRate.value = Math.round(learning_rate / LR_MAX * 100); 
learning_rate_slider.innerHTML = "Learning rate = " + learning_rate; // Display the default slider value

// Update the current slider value (each time you drag the slider handle)
sliderLearningRate.oninput = function() {
    learning_rate = this.value /100 * LR_MAX;
    learning_rate_slider.innerHTML = "Learning rate = " + learning_rate.toFixed(2);
}

/*
--------------------------------------------
BATCH SIZE CONTROL SLIDER
--------------------------------------------
*/
// Note: since the slider is restricted to whole numbers, we have to convert from 0 to 100:

var sliderBatchSize = document.getElementById("slideBatchSize");
var batch_size_slider = document.getElementById("slideBatchSizeLabel");
sliderBatchSize.value = Math.round(batchsize / nData * 100); 
batch_size_slider.innerHTML = "Batch size = " + batchsize; // Display the default slider value

// Update the current slider value (each time you drag the slider handle)
sliderBatchSize.oninput = function() {
    batchsize = Math.round(this.value /100 * nData);
    if (batchsize < 1) {batchsize = 1;}
    batch_size_slider.innerHTML = "Batch size = " + batchsize;
    indices_sgd = []
}

/*
--------------------------------------------
INITIATE THE PLOTS
--------------------------------------------
*/
var state = new_scenario();

update_target_line();
update_hypothesis_line();
update_training_circles();

selected_weight_index = get_nearest_x(w_hat);
select_weight(selected_weight_index);

update_error_line();

