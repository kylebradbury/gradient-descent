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
// [ ] Can delete the training data points
// [ ] Add legend
// [ ] Calculate gradient
// [ ] Move in the direction of gradient (batch)
// [ ] Take step with SGD
// [ ] Checkbox to show/hide target model
// [ ] Checkbox to show/hide target error point
// [ ] Checkbox to color the dots (or not)
// [ ] Button to turn on plot for the hypothesis function
// [ ] Change the target function
// [ ] When performing SGD, show the points that are selected as part of the batch
//     and the corresponding error function just for that subset

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
        // return (w*x - 1.5)**2
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
    let y = x.map(v => f(v)) ;
    
    return make_d3_ready({x:x, y:y}) ;
}

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

// Convert the arrays into arrays of objects with (x,y) pairs
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
    let sample_indices = get_next_batch(),
        N = sample_indices.length;
    sample_indices.forEach(function(i){
        d = state.data[i];
        grad = grad + f_grad(d,w);
    })
    let full_gradient = (2 / N) * grad;
    // console.log("G = " + full_gradient);
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

function get_range(N) {
    let list = [];
    for (let i = 0; i < N; i++) {
        list.push(i);
    }
    return list;
}

function draw_indices(N) {
    let batch = [];
    for (let i = 0; i < N; i++) {
        // If there are no more indices, refill them
        let nAvailableIndices = indices_sgd.length;
        if (nAvailableIndices == 0) {
            indices_sgd = get_range(state.data.length);
            nAvailableIndices = indices_sgd.length;
        }
        // Draw one value and remove it from the index
        let randomIndex = Math.floor(Math.random()*nAvailableIndices); 
        cindex = indices_sgd.splice(randomIndex,1)[0];
        batch.push(cindex);
    }
    return batch;
}

function sgd_update() {
    // Get the gradient
    let grad = gradient() ;

    // Calculate new value for w
    // console.log(-learning_rate * grad);
    w_hat = w_hat - learning_rate * grad;

    // Keep the selection within bounds on the plot
    if (w_hat < WMIN) {
        w_hat = WMIN ;
    } else if (w_hat > WMAX) {
        w_hat = WMAX ;
    }

    // Reduce the learning rate over time
    learning_rate = 0.9999*learning_rate;
    console.log(learning_rate);
}

function adam_update() {
    // Get the gradient
    let g = gradient() ;
    let mt = b1 * m + (1-b1)*g;
    let vt = b2 * v + (1-b2)*g**2;
    t = t + 1 ;
    mt = mt/(1-b1**t) ;
    vt = vt/(1-b2**t) ;
    w_hat = w_hat - learning_rate*mt/(Math.sqrt(vt) + 10**(-8));

    // Keep the selection within bounds on the plot
    if (w_hat < WMIN) {
        w_hat = WMIN ;
    } else if (w_hat > WMAX) {
        w_hat = WMAX ;
    }

    console.log(w_hat);

    // Update parameters for next iteration
    m = mt;
    v = vt;
}

// Define variables that will not be changed interactively
let XMAX = Math.PI,
    YMIN = -1.5, 
    YMAX = 1.5,
    WMIN = 0,
    WMAX = 30,
    EMIN = 0,
    EMAX = 2.5,
    NMAX = 1000,
    nTarget = 601,
    nWeights = 601,
    true_weight = 5,
    NOISE_MAX = 0.5,
    LR_MAX = 10,
    f = f_generator(true_weight);

// Define parameters that may change interactively
let nData = 500,
    w_hat = 10,
    w_plot = 10,
    selected_weight_index = [],
    noise_std = 0.25,
    hypothesis_locked = false,
    indices_sgd = [],
    sgd_mode = 'minibatch',
    batchsize = 10
    learning_rate = 0.5,
    sgd_timer_interval = [],
    interval_set = false;

// Define adam parameters
let m = 0,
    v = 0,
    t = 0,
    b1 = 0.99,
    b2 = 0.99999;

function adam_reset() {
    m = 0;
    v = 0;
    t = 0;
}

// Parameters for plots
var w = 400,
    h = 400,
    margin = { top: 40, right: 20, bottom: 30, left: 40 };
let radius = 5;

function new_scenario() {
    let target = gen_function_data(f,nTarget,XMAX),
        // data = gen_corrupted_data(f,nData,XMAX,noise_std),
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
        .call(d3.drag()
                  .on("start", dragstarted)
                  .on("drag", dragged)
                  .on("end", dragended)
                  )
        .raise();
        // .on("mouseover", handleMouseOver)
        // .on("mouseout", handleMouseOut)
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


function handleMouseOver(d, i) {  // Add interactivity
    d3.select(this)
        .attr("r", radius * 1.5)

}

function handleMouseOut(d, i) {
    d3.select(this)
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

function update_error_line() {
    svg_ep.selectAll("path.line").remove()

    svg_ep.append("path")
        .attr("class", "line")
        .attr("d", valueline_ep(state.error_data));

    update_error_point()
}

// Create a single circle for highlighting selected point
svg_ep.append("circle")
    .attr("id","error_point")
    .on("click", error_plot_clicked);

/*
--------------------------------------------
MOUSE HOVER INTERACTION FOR ERROR PLOT
--------------------------------------------
*/

function error_plot_clicked() {
    hypothesis_locked = !hypothesis_locked;
}

// function mousemoved() {
//     if (hypothesis_locked) {return;}
//     var coord = d3.mouse(this);
//     selected_weight_index = get_nearest_x(coord[0], true);

//     // remove_existing_highlight();
//     select_weight(selected_weight_index);
// }

// function select_weight(index) {
//     let d = state.error_data[index];
//     w_hat = state.weights[index];
//     d3.select('#error_point')
//         .attr("fill", "orange")
//         .attr("r", radius)
//         .attr("cx", xScale_ep(d.x))
//         .attr("cy", yScale_ep(d.y))
//         .attr('weight_index',index)
//         .raise();
//     recalculate_hypothesis();
//     update_hypothesis_line();
//     recalculate_pointwise_error();
//     update_training_circles();
// }

function mousemoved() {
    if (hypothesis_locked) {return;}
    var coord = d3.mouse(this);
    w_hat = xScale_ep.invert(coord[0]);
    selected_weight_index = get_nearest_x(w_hat);

    // remove_existing_highlight();
    select_weight(selected_weight_index);
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
    recalculate_hypothesis();
    update_hypothesis_line();
    recalculate_pointwise_error();
    update_training_circles();
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

// function get_nearest_x(x,scale) {
//     var mindist = 10e6;
//     var dist = [];
//     var index = [];
//     if (scale) {
//         x = xScale_ep.invert(x);    
//     } 
//     state.error_data.forEach(function(d,i) {
//         dist = Math.abs(d.x - x);
//         if (dist < mindist) {
//             index = i;
//             mindist = dist ;
//         }
//     })
//     return index;
// }

function update_for_sgd() {
    sgd_update()
    // adam_update();
    recalculate_hypothesis();
    update_hypothesis_line();
    recalculate_pointwise_error();
    update_training_circles();
    selected_weight_index = get_nearest_x(w_hat);
    select_weight(selected_weight_index);
    // console.log('w = ' + w_hat);
}


function autoupdate_for_sgd() {
    if (interval_set == true) {
        window.clearInterval(sgd_timer_interval);
        interval_set = false;
    } else {
        sgd_timer_interval = window.setInterval(update_for_sgd,10) ;    
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

