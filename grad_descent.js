/*
Gradient descent explorer
Author: Kyle Bradbury
Idea adapted from: http://bl.ocks.org/WilliamQLiu/76ae20060e19bf42d774
*/


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

function f_generator(w) {
    return function(x) {
        return Math.sin(w*x) ; 
    }
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

function add_noise(x,std_noise) {
    return x.map(x => x + randn()*std_noise) ;
}

// Create a representation of the true function
function gen_function_data(f,n,xmax) {
    let x = linspace(0,xmax,n) ;
    let y = x.map(v => f(v)) ;
    return make_d3_ready({x:x, y:y}) ;
}

// Create the training data
function gen_corrupted_data(f,N,xmax,noise_std) {
    let x = rand_array(xmax,N) ;
    x.sort();
    let y = x.map(v => f(v)) ;
    y = add_noise(y, noise_std) ;
    return make_d3_ready({x:x, y:y}) ;
}

function make_d3_ready(data) {
    let d3data = []; 
    for (let i = 0; i < data.x.length; i++) {
        d3data.push({x:data.x[i], y:data.y[i]}) ;
    }
    return d3data;
}

// Define variables that will not be changed interactively
let XMAX = Math.PI,
    YMIN = -1.5, 
    YMAX = 1.5,
    WMIN = 0,
    WMAX = 30,
    EMIN = 0,
    EMAX = 2.5,
    nTarget = 200,
    nWeights = 601,
    true_weight = 5,
    f = f_generator(true_weight);

// Define parameters that may change interactively
let nData = 9,
    w_hat = 6,
    noise_std = 0.25;
    
// let target = gen_function_data(f,nTarget,XMAX),
//     data = gen_corrupted_data(f,nData,XMAX,noise_std),
//     weights = linspace(WMIN,WMAX,nWeights),
//     error_data = mse(weights,data);

function new_scenario() {
    let target = gen_function_data(f,nTarget,XMAX),
        data = gen_corrupted_data(f,nData,XMAX,noise_std),
        weights = linspace(WMIN,WMAX,nWeights),
        error_data = mse(weights,data),
        hypothesis = [] ;
        if (w_hat !== '') {
            hypothesis = gen_function_data(f_generator(w_hat),nTarget,XMAX)  
        }
        
    return {target:target, 
            data:data, 
            weights:weights, 
            error_data:error_data,
            hypothesis:hypothesis};
}

var state = new_scenario();

function refresh() {
    state = new_scenario();
    update_error_line();
    update_target_line();
    update_hypothesis_line();
    update_training_circles();  
}

// [X] Button to generate new data
// [X] Slider to adjust number of data points
// Button to turn on plot for the hypothesis function
// Ability to click a point along the Error plot and display the hypothesis function
// Ability to move the training data points
// Ability to delete the training data points
// Add legend
// Calculate gradient
// Move in the direction of gradient (batch)
// Take step with SGD
// Create slider to adjust noise level

/*
--------------------------------------------
PLOTTING FUNCTION
--------------------------------------------
*/
var w = 400,
    h = 400,
    margin = { top: 40, right: 20, bottom: 30, left: 40 };
let radius = 5;

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
        .attr("r", radius);
}

update_target_line();
update_hypothesis_line();
update_training_circles();

// svg.append("rect")
//     .attr("class", "rect-for-mouseover")
//     .attr("width", w)
//     .attr("height", h)
//     .on("mousemove", mousemoved);

/*
--------------------------------------------
CLICKING AND DRAGGING POINTS
--------------------------------------------
*/



/*
--------------------------------------------
PLOTTING ERROR
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

function update_error_line() {
    svg_ep.selectAll("path.line").remove()

    svg_ep.append("path")
        .attr("class", "line")
        .attr("d", valueline_ep(state.error_data));
}

update_error_line();

// svg_ep.selectAll("circle")
//     .data(error_data)
//     .enter()
//     .append("circle")
//     .attr("cx", function(d) { return xScale_ep(d.x); })
//     .attr("cy", function(d) { return yScale_ep(d.y); })
//     .attr("r", radius);  // Get attributes from circleAttrs var


/*
--------------------------------------------
Controls
--------------------------------------------
*/

var sliderNumSamples = document.getElementById("slideNumSamples");
var output = document.getElementById("slideNumSamplesLabel");
output.innerHTML = "Number of training datapoints = " + sliderNumSamples.value; // Display the default slider value

// Update the current slider value (each time you drag the slider handle)
sliderNumSamples.oninput = function() {
    nData = this.value;
    output.innerHTML = "Number of training datapoints = " + nData;
    refresh();
}


