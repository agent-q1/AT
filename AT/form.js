var http = require('http');
var url = require('url');
var fs = require('fs');
var express =require('express');
var bodyParser=require('body-parser'); 
var urlencodedParser = bodyParser.urlencoded({extended: true});

var app = express();
app.use(express.static('public'));

app.get('/',function(req,res){
    res.send('Submit your name');
})

app.get('/about',function(req,res){
    res.sendFile(__dirname + "/" + "about.html");
})

app.get('/process_get',function(req,res){
    console.log(req.query.coordinates);
    fs.writeFile('c.txt',req.query.coordinates,function(err){
        if(err)
            throw err;
            console.log('updated');
    })
})
app.get('/image-1.jpg', (req, res) => {
    res.sendFile(path.join(__dirname, "./image-1.jpg"));
  });
var server= app.listen(8081,function(){
    var port = server.address().port;
    console.log("%s",port);
}) 