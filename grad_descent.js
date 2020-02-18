/*
Interactive gradient descent explorer
Author: Kyle Bradbury
*/

// TODO:
// [ ] Make buttons change color when toggled on/off
// [ ] Add more interesting functions
// [ ] Cut off points at axes (training data)
// [ ] Resize axes to fit all the data

// May be unnecessary:
// [ ] Make learning rate decay a function of epoch
// [ ] Checkbox to show/hide target error point
// [ ] Slider to adjust target weight
// [ ] Can delete the training data points
// [ ] Add convergence criterion based on last 10 batches (if the change in x < delta for 10 consecutive batches)
// [ ] Checkbox to color the dots (or not)

// Completed:
// [X] Add legend
// [X] Have gradient appear on initial load
// [X] Fix toggle for target function (function automatically reappears when new data is drawn)
// [X] Make the learning rate slider log-scaled
// [X] Make sure gradients and batch error appears for all replots / function changes
/*
--------------------------------------------
INITIALIZATIONS
--------------------------------------------
*/
// Define variables that will not be changed interactively
let EPOCH_MIN = 0,
    EPOCH_MAX = 400,
    NTARGET = 601,
    NWEIGHT = 601,
    NOISE_MAX = 0.5,
	LR_MAX = 10,
	LR_MIN = 0.001,
    MAX_SCREEN_WIDTH = 768;

// Define parameters that may change interactively
let XMAX = Math.PI,
	XMIN = 0,
    YMIN = -1.5, 
    YMAX = 1.5,
    WMIN = -2,
    WMAX = 2,
    EMIN = 0,
    EMAX = 2.5*10,
    NMAX = 200,
	true_weight = 0,
	global_min_weight = [],
	nData = 100,
	w_hat = 10,
	w_hat_new = 10,
	w_hat_old = 10,
	gradient_step = 0,
    selected_weight_index = [],
    noise_std = 0.25,
    hypothesis_locked = false,
    indices_sgd = [],
    sgd_mode = 'minibatch',
    batchsize = 4,
	learning_rate = 0.5,
	learning_rate_stored = 0.5,
    learning_rate_decay = 0.9999,
    sgd_timer_interval = [],
    interval_set = false,
    epoch = 0,
    full_gradient = 0,
    batch_indices = [],
    batch_error = [],
    show_target_function = false,
	show_batch_error = false,
	show_minibatch = false,
    show_gradient_line = false,
	show_learning_curve = true,
	show_true_weight_line = true,
	show_legend = true,
	changed_batch_size = false,
    error_by_epoch = [],
    error_within_epoch = [],
    f_target = [],
    f_hypothesis = [],
    f_grad = [],
    f_type_target = 'sin',
    f_type_hypothesis = 'sin',
    state = [];

// Parameters for plots
let margin = { top: 10, right: 20, bottom: 30, left: 40 },
    radius = 5,
    w,h;

function get_new_plot_sizes() {
	if (window.innerWidth <= MAX_SCREEN_WIDTH) {
		w = window.innerWidth * 0.9;
		h = window.innerHeight / 3 * 0.9;
		h_lc = h;
	} else {
		w = window.innerWidth/2 * 0.85,
	    h = window.innerHeight*2/3 * 0.9;
	    h_lc = h/2;
	}
}

get_new_plot_sizes();

/*
--------------------------------------------
OPTIONS FOR FUNCTIONS
--------------------------------------------
*/

let function_options = {
	parabola: {
		func: function(w) {
		    a = -1.007;
		    b = 1.4167;
		    return function(x) {
		        return a + (x - b - w)**2 ; 
	    	}
	    },
	    grad: function(d,w) {
    		return 4*(d.y - a - (d.x - b - w)**2)*(d.x - b - w);
		},
		XMAX: Math.PI,
		XMIN: 0,
	    YMIN: -1.5, 
	    YMAX: 1.5,
	    WMIN: -2,
	    WMAX: 2,
	    EMIN: 0,
		EMAX: 40,
	    true_weight: 0,
	},
	sin: {
		func:function(w) {
			return function(x) {
				return Math.sin(w*x) ;
			} 
		},
		grad: function(d,w) {
			return (Math.sin(w*d.x) - d.y)*Math.cos(w*d.x)*d.x;
		},
		XMAX: Math.PI,
		XMIN: 0,
	    YMIN: -1.5, 
	    YMAX: 1.5,
		WMIN: 0,
    	WMAX: 20,
    	EMIN: 0,
    	EMAX: 2.5,
    	true_weight: 5,
	},
	cos: {
		func:function(w) {
			return function(x) {
				return Math.cos(w*x) ;
			} 
		},
		grad: function(d,w) {
			return -(Math.cos(w*d.x) - d.y)*Math.sin(w*d.x)*d.x;
		},
		XMAX: Math.PI,
		XMIN: 0,
	    YMIN: -1.5, 
	    YMAX: 1.5,
		WMIN: -1,
    	WMAX: 20,
    	EMIN: 0,
    	EMAX: 2.5,
    	true_weight: 5,
	},
}

let f_mapping = {
	'sin':function_options.sin,
	'parabola':function_options.parabola,
	'cos':function_options.cos,
};

/*
--------------------------------------------
CODE FOR COMPUTATION
--------------------------------------------
*/

let Engine = function(){

	function select_functions(target,hypothesis) {
		set_options_for_function(f_mapping[target],f_mapping[hypothesis]);
	}

	function set_options_for_function(f_tar, f_hyp) {
		f_target = f_tar.func;
		f_hypothesis = f_hyp.func;
		f_grad = f_hyp.grad;
		XMIN = f_hyp.XMIN;
		XMAX = f_hyp.XMAX;
	    YMIN = f_hyp.YMIN;
	    YMAX = f_hyp.YMAX;
	    WMIN = f_hyp.WMIN;
	    WMAX = f_hyp.WMAX;
	    EMIN = f_hyp.EMIN;
	    EMAX = f_hyp.EMAX;
	    true_weight = f_tar.true_weight; // Only applicable when f_tar==f_hyp
	}

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
	        return sum_squared_error(data,f_hypothesis(w)) / N ;
	    })
	    return make_d3_ready({x:weights, y:err}) ;
	}

	function sum_squared_error(data,f) {
	    return data.reduce(function(tot,d){
	        return tot + Math.pow(f(d.x) - d.y,2) ;
	    }, 0)
	}

	function save_current_error() {
		function get_current_error(weight,data) {
	    	let N = data.length;
	    	return sum_squared_error(data,f_hypothesis(weight)) / N ;
		}

	    err = get_current_error(w_hat,state.data);
	    error_within_epoch.push(err);
	}

	function square_error(data,f) {
	    return data.map(function(d){
	        return Math.pow(f(d.x) - d.y,2) ;
	    })
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

	function rand_array(range,N,minval) {
	    let rand_vals = [];
	    for (let i=0; i < N; i++) {
	        rand_vals.push(rand()*range+minval);
	    }
	    return rand_vals;
	}

	// Create a representation of the true function at equally spaced x values
	function gen_function_data(f,n,xmin,xmax) {
	    let x = linspace(xmin,xmax,n) ;
	    let y = x.map(v => f(v)) ;
	    return make_d3_ready({x:x, y:y}) ;
	}

	// Create the noise-free data but with random values for x
	function gen_function_data_random_x(f,N,xmax,xmin) {
	    let x = rand_array(xmax-xmin,N,xmin) ;
	    x.sort();
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
	        d = [];
	    
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
			if (!changed_batch_size) {
				new_epoch() ;
			}
	        batch_indices = get_range(N);
	    } else if (sgd_mode == 'minibatch') {
	        // Check if the pool is empty: if so, refill it
			batch_indices = draw_indices(batchsize) ;
			// console.log(batchsize);
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
	            if (!changed_batch_size) {
					new_epoch() ;
				}
	        }
	        // Draw one value and remove it from the index
	        let randomIndex = Math.floor(Math.random()*nAvailableIndices); 
	        cindex = indices_sgd.splice(randomIndex,1)[0];
	        batch.push(cindex);
	    }
	    return batch;
	}

	function new_epoch() {
	    if ((epoch <= EPOCH_MAX) && (epoch !=0)) {
	        let epoch_mean_error = mean(error_within_epoch) ;
	        error_by_epoch.push({x:epoch-1,y:epoch_mean_error});
	        error_within_epoch = [];
	        PlotLearningCurve.update_learning_curve();
		}
		epoch++;
		console.log(epoch);
	}

	function mean(x) {
	    sumx = x.reduce(function(acc,val){
	        return acc + val;
	    },0)
	    return sumx / x.length
	}

	function clear_batches() {
		batch_indices = [];
    	batch_error = [];
	}

	function sgd_compute() {
		// Get the gradient
	    let grad = gradient() ;

	    // Calculate new value for w
		gradient_step = - learning_rate * grad;
	    w_hat_new = w_hat + gradient_step;

	    // Keep the selection within bounds on the plot
	    if (w_hat_new < WMIN) {
	        w_hat_new = WMIN ;
	    } else if (w_hat_new > WMAX) {
	        w_hat_new = WMAX ;
	    }
	}

	function sgd_update() {
	    // Save the error to an array for learning curves
	    save_current_error();
		sgd_compute();

		w_hat = w_hat_new;

	    // Reduce the learning rate over time
		learning_rate = learning_rate_decay*learning_rate;
		
		// Draw the next batch
		get_next_batch();
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

	function update_min_weight_val() {
		let minval = 10e6;
		state.error_data.forEach(function(d) {
			if (d.y < minval) {
				minval = d.y;
				global_min_weight = d.x;
			}
		});
	}

	// Create a new set of all values for the data
	function new_scenario() {
		select_functions(f_type_target,f_type_hypothesis);
	    let target_function = f_target(true_weight),
	        predicted_function = f_hypothesis(w_hat);

	    let target = gen_function_data(target_function,NTARGET,XMIN,XMAX),
	        noisefree_data = gen_function_data_random_x(target_function,nData,XMAX,XMIN),
	        noise = gen_noise(nData),
	        data = corrupt_data(noisefree_data, noise);
	        weights = linspace(WMIN,WMAX,NWEIGHT),
	        error_data = mse(weights,data),
			pointwise_error = square_error(data, predicted_function)
	        hypothesis = [] ;
	        if (w_hat !== '') {
	            hypothesis = gen_function_data(predicted_function,NTARGET,XMIN,XMAX);
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

	// Error associated with each possible weight value
	function recalculate_error_function_E_of_w() {
		recalculate_pointwise_error();
		state.error_data = mse(state.weights,state.data);
		update_min_weight_val();
	}

	// Noise to be added to the target data
	function recalculate_noise() {
	    state.data = corrupt_data(state.noisefree_data, state.noise);
	    state.error_data = mse(state.weights,state.data)
	}

	// Target values
	function recalculate_hypothesis() {
	    state.hypothesis = gen_function_data(f_hypothesis(w_hat),NTARGET,XMIN,XMAX)  
	}

	// Error associated with each individual point in the training set
	function recalculate_pointwise_error() {
	    state.pointwise_error = square_error(state.data, f_hypothesis(w_hat))
	}

	// Make some of these functions available outside the namespace
	return {
		sgd_update:sgd_update,
		sgd_compute:sgd_compute,
		get_batch_error:get_batch_error,
		new_scenario:new_scenario,
		recalculate_error_function_E_of_w:recalculate_error_function_E_of_w,
		recalculate_noise:recalculate_noise,
		recalculate_hypothesis:recalculate_hypothesis,
		recalculate_pointwise_error:recalculate_pointwise_error,
		update_min_weight_val:update_min_weight_val,
		clear_batches:clear_batches,
		get_next_batch:get_next_batch,
	}

}(); // End Engine (code for computation)

/*
--------------------------------------------
PLOT DATA, TARGET, AND ESTIMATE
--------------------------------------------
*/
let PlotData = function() {
	let xScale, yScale, xAxis, yAxis, valueline, svg

	function draw() {
		d3.selectAll('.plotdata').remove();

		// We're passing in a function in d3.max to tell it what we're maxing (x value)
		xScale = d3.scaleLinear()
		    .domain([XMIN, XMAX])
		    .range([margin.left, w - margin.right]);  // Set margins for x specific

		// We're passing in a function in d3.max to tell it what we're maxing (y value)
		yScale = d3.scaleLinear()
		    .domain([YMIN, YMAX])
		    .range([h - margin.bottom, margin.top]);  // Set margins for y specific

		// Add a X and Y Axis (Note: orient means the direction that ticks go, not position)
		xAxis = d3.axisBottom(xScale);
		yAxis = d3.axisLeft(yScale);

		// Define the line
		valueline = d3.line()
		    .x(function(d) { return xScale(d.x); })
		    .y(function(d) { return yScale(d.y); });

		// Define the color scale
		error_color_scale = d3.scaleLinear()
		    .domain([0,2])
		    .range(["orange", "black"])

		svg = d3.select("#plot_function")
		    .append("svg")
		    .attr("width", w)
		    .attr("height", h)
		    .attr("class",'plotdata');

		// Adds X-Axis as a 'g' element
		svg.append("g").attr("class", "axis")
		  .attr("transform", "translate(" + [0, h - margin.bottom] + ")")  // Translate just moves it down into position (or will be on top)
		  .call(d3.axisBottom(xScale));  // Call the xAxis function on the group

		// Adds Y-Axis as a 'g' element
		svg.append("g").attr("class", "axis")
		  .attr("transform", "translate(" + [margin.left, 0] + ")")
		  .call(yAxis);  // Call the yAxis function on the group

		// Add axis labels
		// X-Label
		svg.append("text")
		    .attr("class", "x label")
		    .attr("text-anchor", "end")
		    .attr("alignment-baseline", "bottom")
		    .attr("x", w)
		    .attr("y", h)
		    .text("x");

		// Y-Label
		svg.append("text")
		    .attr("class", "y label")
		    .attr("text-anchor", "middle")
		    .attr("y", 6)
		    .attr("dy", ".75em")
		    .attr("x", -margin.top)
		    .attr("transform", "rotate(-90)")
			.text("f(x,w)");
		
		// Legend
		svg.append("svg:image")
				.attr("class","legend")
				.attr("xlink:href", "./img/legend1.png")
				.attr("x", w-160)
				.attr("y", h-170)
				.attr("preserveAspectRatio","xMaxYMax")
				.attr("width", 150)
				.attr("height", 140)
				.attr("visibility", function(){
					if (show_legend) {return "visible"}
					else {return "hidden"}
				});

	} // draw()

	draw();

	function update_legend() {
		svg.selectAll(".legend").attr("visibility",function() {
			if (show_legend) {return "visible"}
			else {return "hidden"}
		})
	}

	function update_target_line() {
	    svg.selectAll("path.targetline").remove()

	    svg.append("path")
	        .attr("class", "targetline")
	        .attr("id", "target")
			.attr("d", valueline(state.target))
			.attr("visibility", function(){
				if (show_target_function) {
					return "visible";
				} else {
					return "hidden";
				}
			});
	}

	function update_hypothesis_line() {
	    svg.selectAll("path.hypothesisline").remove()

	    svg.append("path")
	        .attr("class", "hypothesisline")
	        .attr("id", "hypothesis")
	        .attr("d", valueline(state.hypothesis));
	}

	function update_training_circles() {
	    var training_circles = svg.selectAll("circle.data")
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
	            if (batch_indices.includes(i) && show_minibatch) {
	                return "batch-sample data";
	            } else {
	                return "data";
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
	    d3.select(this)
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
	    if (d.x < XMIN) {d.x = XMIN;}
	    if (d.x > XMAX) {d.x = XMAX;}
	    if (d.y < YMIN) {d.y = YMIN;}
	    if (d.y > YMAX) {d.y = YMAX;}

	    // Redraw the error functions based on the new data
		Engine.recalculate_error_function_E_of_w();
		Engine.get_batch_error();
		PlotError.update_error_line();
		PlotError.update_true_weight_line();
		PlotError.update_batch_error_line();

		Engine.sgd_compute();
		PlotError.update_gradient_line();


	    d3.select(this)
	        .attr("cx", xScale(d.x))
	        .attr("cy", yScale(d.y))
	        .attr("r", radius * 1.5)
	        .attr("fill", function(d,i){ 
	            return error_color_scale(state.pointwise_error[index]); 
			})
			.attr('class', function(d,i) {
	            if (batch_indices.includes(+index) && show_minibatch) {
	                return "batch-sample";
	            } else {
	                return "";
	            }
	        })
	        .raise();
	}

	function dragended(d) {
	    d3.select(this)
	        .attr("r", radius);
	}

	return {
		draw:draw,
		update_target_line:update_target_line,
		update_hypothesis_line:update_hypothesis_line,
		update_training_circles:update_training_circles,
		dragstarted:dragstarted,
		dragged:dragged,
		dragended:dragended,
		update_legend:update_legend,
	}
}(); // PlotData 

/*
--------------------------------------------
PLOT ERROR
--------------------------------------------
*/
let PlotError = function() {
	let xScale_ep, yScale_ep, xAxis_ep, yAxis_ep, valueline_ep, svg_ep

	function draw() {
		// Remove any old plots present
		d3.selectAll(".ploterror").remove();

		// We're passing in a function in d3.max to tell it what we're maxing (x value)
		xScale_ep = d3.scaleLinear()
		    .domain([WMIN, WMAX])
		    .range([margin.left, w - margin.right]);  // Set margins for x specific

		// We're passing in a function in d3.max to tell it what we're maxing (y value)
		yScale_ep = d3.scaleLinear()
		    .domain([EMIN, EMAX])
		    .range([h - margin.bottom, margin.top]);  // Set margins for y specific

		// Add a X and Y Axis (Note: orient means the direction that ticks go, not position)
		xAxis_ep = d3.axisBottom(xScale_ep);
		yAxis_ep = d3.axisLeft(yScale_ep);

		// Define the line
		valueline_ep = d3.line()
		    .x(function(d) { return xScale_ep(d.x); })
		    .y(function(d) { return yScale_ep(d.y); });

		svg_ep = d3.select("#plot_error")
		    .append("svg")
		    .attr("width", w)
		    .attr("height", h)
		    .attr("class","ploterror");

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

		// Draw the line that corresponds to the global minimum (over the visible domain)
		svg_ep.append("path")
		        .attr("class", "true-weight-line")
		        .attr("d", valueline_ep([{x:global_min_weight, y:EMIN},{x:global_min_weight, y:EMAX}]));

		// Create a single circle for highlighting selected point
		old_point = svg_ep.append("circle")
		    .attr("id","old-error_point");

		svg_ep.append("circle")
		    .attr("id","error_point")
		    .on("click", error_plot_clicked);

		svg_ep.append("path")
		        .attr("class", "gradient")
		        .attr("d", valueline_ep([
		            {x:w_hat, y:yScale_ep.invert(d3.select('#error_point').attr('cy'))},
					{x:w_hat + gradient_step, y:yScale_ep.invert(d3.select('#error_point').attr('cy'))}]));
					
		if (show_batch_error) {
			svg_ep.append("path")
				.attr("class", "batch-error-line")
				.attr("d", valueline_ep(batch_error));
		}

		// Legend
		svg_ep.append("svg:image")
				.attr("class","legend")
				.attr("xlink:href", "./img/legend2.png")
				.attr("x", w-160)
				.attr("y", 0)
				.attr("preserveAspectRatio","xMaxYMin")
				.attr("width", 150)
				.attr("height", 150)
				.attr("visibility",function() {
					if (show_legend) {return "visible"}
					else {return "hidden"}
				});
	} // draw()

	draw();
	
	function update_legend() {
		svg_ep.selectAll(".legend").attr("visibility",function() {
			if (show_legend) {return "visible"}
			else {return "hidden"}
		})
	}

	function error_plot_clicked() {
	    hypothesis_locked = !hypothesis_locked;
	}

	function update_error_line() {
	    svg_ep.selectAll("path.line").remove()

	    svg_ep.append("path")
	        .attr("class", "line")
	        .attr("d", valueline_ep(state.error_data));

	    update_error_point()
	}

	function update_true_weight_line() {
		svg_ep.selectAll("path.true-weight-line").remove()

		if (show_true_weight_line) {
			// Draw the line that corresponds to the global minimum (over the visible domain)
			svg_ep.append("path")
				.attr("class", "true-weight-line")
				.attr("d", valueline_ep([{x:global_min_weight, y:EMIN},{x:global_min_weight, y:EMAX}]));
		}
	}

	function update_batch_error_line() {
	    svg_ep.selectAll("path.batch-error-line").remove()

	    if (show_batch_error) {
	        svg_ep.append("path")
	            .attr("class", "batch-error-line")
	            .attr("d", valueline_ep(batch_error));
	    }
	}

	function update_gradient_line() {
	    svg_ep.selectAll("path.gradient").remove()

	    if (show_gradient_line) {
	        svg_ep.append("path")
	            .attr("class", "gradient")
	            .attr("d", valueline_ep([
	                {x:w_hat, y:yScale_ep.invert(d3.select('#error_point').attr('cy'))},
	                {x:w_hat + gradient_step, y:yScale_ep.invert(d3.select('#error_point').attr('cy'))}]));
	    }
	}

	// Mouse hover interaction
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
	    Engine.recalculate_hypothesis();
		PlotData.update_hypothesis_line();
	    Engine.recalculate_pointwise_error();
		PlotData.update_training_circles();
		Engine.sgd_compute();
		update_gradient_line();
	}

	function select_weight(index) {
		let d = state.error_data[index];

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

	return {
		svg:svg_ep,
		xScale:xScale_ep,
		yScale:yScale_ep,
		xAxis:xAxis_ep,
		yAxis:yAxis_ep,
		valueline:valueline_ep,
		update_error_line:update_error_line,
		update_batch_error_line:update_batch_error_line,
		update_gradient_line:update_gradient_line,
		update_true_weight_line:update_true_weight_line,
		update_legend:update_legend,
		select_weight:select_weight,
		draw:draw,
	}
}(); // PlotError

/*
--------------------------------------------
LEARNING CURVE PLOT
--------------------------------------------
*/
let PlotLearningCurve = function() {
	let xScale_lc, yScale_lc, xAxis_lc, yAxis_lc, valueline_lc, svg_lc;
	
	function draw() {
		d3.selectAll(".learningcurve").remove()

		// We're passing in a function in d3.max to tell it what we're maxing (x value)
		xScale_lc = d3.scaleLinear()
		    .domain([EPOCH_MIN,EPOCH_MAX])
		    .range([margin.left, w - margin.right]);  // Set margins for x specific

		// We're passing in a function in d3.max to tell it what we're maxing (y value)
		yScale_lc = d3.scaleLinear()
		    .domain([EMIN, EMAX])
		    .range([h_lc - margin.bottom, margin.top]);  // Set margins for y specific

		// Add a X and Y Axis (Note: orient means the direction that ticks go, not position)
		xAxis_lc = d3.axisBottom(xScale_lc);
		yAxis_lc = d3.axisLeft(yScale_lc);

		// Define the line
		valueline_lc = d3.line()
		    .x(function(d) { return xScale_lc(d.x); })
		    .y(function(d) { return yScale_lc(d.y); });

		svg_lc = d3.select("#plot_learning_curve")
		    .append("svg")
		    .attr("width", w)
		    .attr("height", h_lc)
		    .attr("class","learningcurve");

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
	} // draw()

	draw();

	function update_learning_curve() {
	    svg_lc.selectAll("path.learning-curve").remove()

	    if (show_learning_curve) {
	        svg_lc.append("path")
	            .attr("class", "learning-curve")
	            .attr("d", valueline_lc(error_by_epoch));
	    }
	}
	return {
		update_learning_curve:update_learning_curve,
		draw:draw,
	}
}(); // PlotLearningCurve

/*
--------------------------------------------
UPDATE FUNCTIONS AND MISC
--------------------------------------------
*/

// Draw new data and recalculate component values
function refresh() {
	state = Engine.new_scenario();
	Engine.update_min_weight_val();
	PlotError.update_error_line();
	PlotError.update_true_weight_line();
    PlotData.update_target_line();
    PlotData.update_hypothesis_line();
	PlotData.update_training_circles();
	update_sliders();
	reset_sgd();
	PlotError.update_batch_error_line();
}

function reset_sgd() {
    learning_rate = learning_rate_stored;
    epoch = 0;
    error_by_epoch = [];
	error_within_epoch = [];
	Engine.clear_batches();
	Engine.get_next_batch();
	PlotError.update_batch_error_line();
	PlotData.update_training_circles();
    PlotLearningCurve.update_learning_curve();
    document.getElementById('learningrate').innerHTML = "Learning Rate = " + learning_rate.toFixed(4) ;
    document.getElementById('weight').innerHTML = "Weight = " + w_hat.toFixed(3) ;
    document.getElementById('epoch').innerHTML = "Epoch = " + epoch ;
    d3.selectAll(".gradient").remove();
}

function update_for_sgd() {
	Engine.sgd_update();
	Engine.sgd_compute();
    Engine.recalculate_hypothesis();
    PlotData.update_hypothesis_line();
    Engine.recalculate_pointwise_error();
    
    selected_weight_index = get_nearest_x(w_hat);
    PlotError.select_weight(selected_weight_index);
    document.getElementById('learningrate').innerHTML = "Learning Rate = " + learning_rate.toFixed(4) ;
    document.getElementById('weight').innerHTML = "Weight = " + w_hat.toFixed(3) ;
    document.getElementById('epoch').innerHTML = "Epoch = " + epoch ;

    
    if (show_batch_error) {
        Engine.get_batch_error();    
    }
    
    PlotError.update_batch_error_line();
	PlotData.update_training_circles();
	
    PlotError.update_gradient_line();
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

function replot_all_plots() {
	get_new_plot_sizes();
	PlotData.draw();
	PlotError.draw();
	PlotLearningCurve.draw();
	PlotData.update_target_line();
	PlotData.update_hypothesis_line();
	PlotData.update_training_circles();

	selected_weight_index = get_nearest_x(w_hat);
	PlotError.select_weight(selected_weight_index);
	PlotError.update_error_line();
	PlotError.update_true_weight_line();
	PlotError.update_gradient_line();
}

window.addEventListener('resize', replot_all_plots);

/*
--------------------------------------------
FUNCTIONS USED FROM THE MAIN USER INTERFACE
--------------------------------------------
*/
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
        buttonlabel.innerHTML = 'Minibatch' ;
    } else if (sgd_mode == 'minibatch') {
        sgd_mode = 'batch';
        buttonlabel.innerHTML = "Batch" ;
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
        Engine.get_batch_error()
    } 
	show_batch_error = !show_batch_error;
	show_minibatch = !show_minibatch;
	PlotError.update_batch_error_line();
	PlotData.update_training_circles();
}

function toggle_gradient() {
    show_gradient_line = !show_gradient_line;
    PlotError.update_gradient_line();
}

function toggle_legend() {
	show_legend = !show_legend;
	PlotData.update_legend();
    PlotError.update_legend();
}

/*
--------------------------------------------
SLIDER CONTROLS AND INTERACTIONS
--------------------------------------------
*/

let Slider = function(config) {
	var slider_element = document.getElementById(config.slider_element);
	var slider_label = document.getElementById(config.slider_label);
	slider_element.step = 0.5;
	if (config.scale == 'linear') {
		slider_element.value = (config.value - config.min_value)  / (config.max_value-config.min_value) * 100; 
	} else if (config.scale = 'log') {
		slider_element.value = (Math.log10(config.value) - Math.log10(config.min_value))  / (Math.log10(config.max_value)-Math.log10(config.min_value)) * 100; 
	}
	slider_label.innerHTML = config.slider_label_name + " = " + config.value.toFixed(config.to_fixed); // Display the default slider value

	// Update the current slider value (each time you drag the slider handle)
	slider_element.oninput = function() {
		if (config.scale == 'linear') {
			config.value = this.value / 100 * (config.max_value-config.min_value) + config.min_value; 
		} else if (config.scale = 'log') {
			config.value = 10**(this.value / 100 * (Math.log10(config.max_value)-Math.log10(config.min_value)) + Math.log10(config.min_value)); 
		}
	    config.setval(config.value);
	    slider_label.innerHTML = config.slider_label_name + " = " + config.value.toFixed(config.to_fixed);
	    config.func();
	}
}

let config_slider_n_training, 
	config_slider_noise,
	config_slider_learning_rate,
	config_slider_batch_size;


function update_sliders(update_n_training=true) {
	config_slider_n_training = {
		slider_element: "slideNumSamples",
		slider_label: "slideNumSamplesLabel",
		slider_label_name: "Number of training datapoints",
		value: nData,
		max_value: NMAX,
		min_value: 1,
		to_fixed: 0,
		scale: 'linear',
		setval: function(value) {
			nData = Math.round(value);
		},
		func: function() {
			indices_sgd = [];
			if (nData < batchsize) {
				batchsize = nData;
			}
			update_sliders(false);
			state = Engine.new_scenario();
			reset_sgd();
			Engine.update_min_weight_val();
			PlotError.update_error_line();
			PlotError.update_true_weight_line();
			PlotData.update_target_line();
			PlotData.update_hypothesis_line();
			PlotData.update_training_circles();
			Engine.get_batch_error();
			PlotError.update_batch_error_line();
		}
	};

	config_slider_noise = {
		slider_element: "slideNoise",
		slider_label: "slideNoiseLabel",
		slider_label_name:"Noise std",
		value: noise_std,
		max_value: NOISE_MAX,
		min_value: 0,
		to_fixed: 3,
		scale: 'linear',
		setval: function(value) {
			noise_std = value;
		},
		func: function() {
			Engine.recalculate_noise();
			Engine.get_batch_error();
    		PlotData.update_training_circles();
			PlotError.update_error_line();
			Engine.sgd_compute();
			PlotError.update_gradient_line();
			PlotError.update_batch_error_line();
		}
	};

	config_slider_learning_rate = {
		slider_element: "slideLearningRate",
		slider_label: "slideLearningRateLabel",
		slider_label_name:"Learning rate",
		value: learning_rate,
		max_value: LR_MAX,
		min_value: LR_MIN,
		to_fixed: 3,
		scale: 'log',
		setval: function(value) {
			learning_rate = value;
			learning_rate_stored = value;
		},
		func: function() {
			Engine.sgd_compute();
			PlotError.update_gradient_line();
		}
	};

	config_slider_batch_size = {
		slider_element: "slideBatchSize",
		slider_label: "slideBatchSizeLabel",
		slider_label_name:"Batch size",
		value: batchsize,
		max_value: nData,
		min_value: 1,
		to_fixed: 0,
		scale: 'linear',
		setval: function(value) {
			batchsize = Math.round(value);
		},
		func: function() {
			indices_sgd = [];
			batch_indices = [];
			changed_batch_size = true;
			Engine.clear_batches();
			Engine.get_next_batch();
			Engine.get_batch_error();
			PlotError.update_batch_error_line();
			changed_batch_size = false;
		}
	};

	if (update_n_training) {
		Slider(config_slider_n_training);
	}
	Slider(config_slider_noise);
	Slider(config_slider_learning_rate);
	Slider(config_slider_batch_size);

}

let FunctionDropdown = function() {
	var dropdown_target = document.getElementById('func_target');
	var dropdown_hypothesis = document.getElementById('func_hypothesis');

	var function_update = function() {
		// Check if either dropdown changed
		if ((dropdown_target.value != f_type_target) || 
			(dropdown_hypothesis.value != f_type_hypothesis)) {

			f_type_target = dropdown_target.value;
			f_type_hypothesis = dropdown_hypothesis.value;
			state = Engine.new_scenario();
			Engine.update_min_weight_val()
			reset_sgd()
			PlotData.draw();
			PlotError.draw();
			PlotLearningCurve.draw();
			PlotData.update_target_line();
			PlotData.update_hypothesis_line();
			PlotData.update_training_circles();

			selected_weight_index = get_nearest_x(w_hat);
			PlotError.select_weight(selected_weight_index);
			PlotError.update_error_line();
			Engine.get_batch_error();
			PlotError.update_batch_error_line();
			update_sliders();
			PlotError.update_gradient_line();
			
		}
	}

	dropdown_target.oninput = function_update;
	dropdown_hypothesis.oninput = function_update;
}();

/*
--------------------------------------------
INITIATE THE PLOTS
--------------------------------------------
*/
get_new_plot_sizes();
state = Engine.new_scenario();
reset_sgd();
Engine.update_min_weight_val()
PlotData.draw();
PlotError.draw();
PlotLearningCurve.draw();
PlotData.update_target_line();
PlotData.update_hypothesis_line();
PlotData.update_training_circles();

selected_weight_index = get_nearest_x(w_hat);
PlotError.select_weight(selected_weight_index);
PlotError.update_error_line();
Engine.get_batch_error();
PlotError.update_batch_error_line();
update_sliders();
Engine.sgd_compute();
PlotError.update_gradient_line();