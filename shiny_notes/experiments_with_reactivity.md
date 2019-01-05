Experiments With Reactivity
================
Giovanni d'Ario

-   [Introduction](#introduction)
-   [Basic App](#basic-app)

Introduction
------------

In this document we try to understand how reactivity works by creating apps of increasing complexity, and testing when and where they fail.

Basic App
---------

The simplest app is the following

``` r
library(shiny)
ui <- fluidPage(
  sliderInput(inputId = "num", label = "Value", min = 1, max = 100, value = 50),
  verbatimTextOutput("value")
)
server <- function(input, output) {
  output$value <- renderText({input$num})
}
shinyApp(ui = ui, server = server)
```

This app works as expected: the `input$num` value is used in a reactive function, and whenever the slider is moved, the printed value changes. What happens if we move `input$num` outside of the reactive function?

``` r
library(shiny)
ui <- fluidPage(
  sliderInput(inputId = "num", label = "Value", min = 1, max = 100, value = 50),
  verbatimTextOutput("value")
)
server <- function(input, output) {
  x <- input$num # This doesn't work
  output$value <- renderText({x})
}
shinyApp(ui = ui, server = server)
```

This app fails, since `x` is not defined as a reactive value, but is still used in a reactive function. What if we define it `reactive`?

``` r
library(shiny)
ui <- fluidPage(
  sliderInput(inputId = "num", label = "Value", min = 1, max = 100, value = 50),
  verbatimTextOutput("value")
)
server <- function(input, output) {
  x <- reactive(input$num) # Now x is reactive
  output$value <- renderText({x()}) # Use x()!!!
}
shinyApp(ui = ui, server = server)
```

This app works fine.
