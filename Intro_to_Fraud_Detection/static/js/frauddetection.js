
$(document).ready(function () {
    // An application can open a connection on multiple namespaces, and
    // Socket.IO will multiplex all those connections on a single
    // physical channel. If you don't care about multiple channels, you
    // can set the namespace to an empty string.
    var namespace = "/fraud";

    // Connect to the Socket.IO server.
    // The connection URL has the following format:
    //     http[s]://<domain>:<port>[/<namespace>]
    var socket_url = location.protocol + "//" + document.domain + ":" + location.port + namespace;
    var socket = io.connect(socket_url);
    console.log("Connected to " + socket_url);

    // Handler for health signal
    socket.on("health_signal", function (msg) {
        console.log("Response from server: " + msg.data + " (note: " + msg.note + ")");
    });

    // Location
    function Location(title, latitude, longitude) {
        this.title = title;
        this.latitude = parseFloat(latitude);
        this.longitude = parseFloat(longitude);
        this.scale = 0.5;
        this.zoomLevel = 5;
    }

    // Placeholder for map locations
    var mapLocations = [];

    // Based on https://www.amcharts.com/demos/custom-html-elements-map-markers/ 
    var map = AmCharts.makeChart("chartdiv", {
        "type": "map",
        "theme": "none",
        "projection": "miller",
        "imagesSettings": {
            "rollOverColor": "#089282",
            "rollOverScale": 3,
            "selectedScale": 3,
            "selectedColor": "#089282",
            "color": "#13564e"
        },
        "areasSettings": {
            "unlistedAreasColor": "#222222" /* change color of the map */
        },
        "dataProvider": {
            "map": "worldLow",
            "images": mapLocations
        }
    });

    // Location updated emitted by the server via websockets
    socket.on("map_update", function (msg) {
        var message = "New event in " + msg.title + " (" + msg.latitude
            + "," + msg.longitude + ")";
        console.log(message);
        var newLocation = new Location(msg.title, msg.latitude, msg.longitude);
        mapLocations.push(newLocation);

        //clear the markers before redrawing
        mapLocations.forEach(function(location) {
          if (location.externalElement) {
            location.externalElement = undefined;
          }
        });

        map.dataProvider.images = mapLocations;
        map.validateData(); //call to redraw the map with new data
    });

    // add events to recalculate map position when the map is moved or zoomed
    map.addListener("positionChanged", updateCustomMarkers);

    // this function will take current images on the map and create HTML elements for them
    function updateCustomMarkers(event) {
        // get map object
        var map = event.chart;

        // go through all of the images
        for (var x in map.dataProvider.images) {
            // get MapImage object
            var image = map.dataProvider.images[x];

            // check if it has corresponding HTML element
            if ('undefined' == typeof image.externalElement)
                image.externalElement = createCustomMarker(image);

            // reposition the element accoridng to coordinates
            var xy = map.coordinatesToStageXY(image.longitude, image.latitude);
            image.externalElement.style.top = xy.y + 'px';
            image.externalElement.style.left = xy.x + 'px';
        }
    }

    // this function creates and returns a new marker element
    function createCustomMarker(image) {
        // create holder
        var holder = document.createElement('div');
        holder.className = 'map-marker';
        holder.title = image.title;
        holder.style.position = 'absolute';

        // maybe add a link to it?
        if (undefined != image.url) {
            holder.onclick = function () {
                window.location.href = image.url;
            };
            holder.className += ' map-clickable';
        }

        // create dot
        var dot = document.createElement('div');
        dot.className = 'dot';
        holder.appendChild(dot);

        // create pulse
        var pulse = document.createElement('div');
        pulse.className = 'pulse';
        holder.appendChild(pulse);

        // append the marker to the map container
        image.chart.chartDiv.appendChild(holder);

        return holder;
    }

    // Interval function that tests message latency by sending a "ping"
    // message. The server then responds with a "pong" message and the
    // round trip time is measured.
    var pingPongTimes = [];
    var startTime;
    window.setInterval(function () {
        startTime = (new Date).getTime();
        socket.emit("my_ping");
    }, 1000);

    // Handler for the "pong" message. When the pong is received, the
    // time from the ping is stored, and the average of the last 30
    // samples is average and displayed.
    socket.on("my_pong", function () {
        var latency = (new Date).getTime() - startTime;
        pingPongTimes.push(latency);
        pingPongTimes = pingPongTimes.slice(-30); // keep last 30 samples
        var sum = 0;
        for (var i = 0; i < pingPongTimes.length; i++) {
            sum += pingPongTimes[i];
        }
        // console.log("Pong sum=" + sum);
        $("#ping-pong").text(Math.round(10 * sum / pingPongTimes.length) / 10);
    });

}); // end jquery $(document).ready(function ()
